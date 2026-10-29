import streamlit as st
import chromadb
from chromadb.utils.embedding_functions import OllamaEmbeddingFunction
import ollama
import logging
import shutil
import sqlite3
import subprocess
import threading
import time
from datetime import datetime, timezone

# 1. Page Configuration
st.set_page_config(page_title="Movie Recommender", layout="centered", page_icon="🎬", initial_sidebar_state="expanded")
st.title("🎬 Movie Recommendation Agent")

# New Chat Button to Clear Conversation History
with st.sidebar:
    st.header("Chat Controls")
    
    # When clicked, this button resets the chat memory to an empty list
    if st.button("➕ Start New Chat"):
        st.session_state.messages = []

# 1.5 Initialize SQLite database for agent telemetry
TELEMETRY_DB = "telemetry.sqlite3"
NVIDIA_SMI = shutil.which("nvidia-smi")

def init_telemetry_db():
    conn = sqlite3.connect(TELEMETRY_DB)
    try:
        with conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS model_calls (
                    id INTEGER PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    model TEXT NOT NULL,
                    call_type TEXT NOT NULL,
                    wall_ms REAL NOT NULL,
                    total_duration_ns INTEGER,
                    prompt_tokens INTEGER,
                    output_tokens INTEGER,
                    prompt_eval_duration_ns INTEGER,
                    eval_duration_ns INTEGER,
                    tokens_per_second REAL,
                    time_to_first_token_ms REAL,
                    gpu_temp_c REAL,
                    gpu_utilization_pct REAL,
                    gpu_metrics_status TEXT,
                    error TEXT,
                    success INTEGER NOT NULL
                )
            """)
            existing_columns = {
                row[1] for row in conn.execute("PRAGMA table_info(model_calls)")
            }
            new_columns = {
                "prompt_eval_duration_ns": "INTEGER",
                "eval_duration_ns": "INTEGER",
                "tokens_per_second": "REAL",
                "time_to_first_token_ms": "REAL",
                "gpu_temp_c": "REAL",
                "gpu_utilization_pct": "REAL",
                "gpu_metrics_status": "TEXT",
                "error": "TEXT",
            }
            for column, data_type in new_columns.items():
                if column not in existing_columns:
                    conn.execute(f"ALTER TABLE model_calls ADD COLUMN {column} {data_type}")
    finally:
        conn.close()

def _field(response, name):
    if isinstance(response, dict):
        return response.get(name)
    return getattr(response, name, None)

class GpuMonitor:
    """Samples NVIDIA GPU temperature and utilization while a model call runs."""

    def __init__(self):
        self.samples = []
        self.status = "nvidia-smi unavailable" if NVIDIA_SMI is None else "no samples"
        self._sampling_failed = False
        self._stop_event = threading.Event()
        self._thread = None

    def _sample(self):
        if NVIDIA_SMI is None or self._sampling_failed:
            return
        try:
            result = subprocess.run(
                [
                    NVIDIA_SMI,
                    "--query-gpu=temperature.gpu,utilization.gpu",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                check=True,
                timeout=2,
            )
            samples = []
            for line in result.stdout.splitlines():
                temperature, utilization = line.split(",")
                samples.append((float(temperature.strip()), float(utilization.strip())))
            if samples:
                self.samples.extend(samples)
                self.status = "available"
            else:
                self.status = "nvidia-smi returned no GPUs"
                self._sampling_failed = True
        except (OSError, subprocess.SubprocessError, ValueError) as exc:
            self.status = f"sampling failed: {exc}"
            self._sampling_failed = True
            logging.warning("Could not sample NVIDIA GPU telemetry: %s", exc)

    def _sample_repeatedly(self):
        while not self._stop_event.is_set():
            self._sample()
            self._stop_event.wait(0.5)

    def start(self):
        self._thread = threading.Thread(target=self._sample_repeatedly, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.5)

    def metrics(self):
        if not self.samples:
            return None, None, self.status
        return (
            max(sample[0] for sample in self.samples),
            max(sample[1] for sample in self.samples),
            self.status,
        )

def log_model_call(
    model,
    call_type,
    started_at,
    response,
    gpu_monitor,
    time_to_first_token_ms=None,
    error=None,
):
    prompt_tokens = _field(response, "prompt_eval_count")
    output_tokens = _field(response, "eval_count")
    eval_duration_ns = _field(response, "eval_duration")
    tokens_per_second = (
        output_tokens * 1_000_000_000 / eval_duration_ns
        if output_tokens is not None and eval_duration_ns
        else None
    )
    gpu_temp_c, gpu_utilization_pct, gpu_status = gpu_monitor.metrics()

    conn = sqlite3.connect(TELEMETRY_DB)
    try:
        with conn:
            conn.execute("""
                INSERT INTO model_calls (
                    created_at, model, call_type, wall_ms, total_duration_ns,
                    prompt_tokens, output_tokens, prompt_eval_duration_ns,
                    eval_duration_ns, tokens_per_second, time_to_first_token_ms,
                    gpu_temp_c, gpu_utilization_pct, gpu_metrics_status, error, success
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                datetime.now(timezone.utc).isoformat(),
                model,
                call_type,
                (time.perf_counter() - started_at) * 1000,
                _field(response, "total_duration"),
                prompt_tokens,
                output_tokens,
                _field(response, "prompt_eval_duration"),
                eval_duration_ns,
                tokens_per_second,
                time_to_first_token_ms,
                gpu_temp_c,
                gpu_utilization_pct,
                gpu_status,
                repr(error) if error is not None else None,
                int(error is None),
            ))
    finally:
        conn.close()

init_telemetry_db()
if NVIDIA_SMI is None:
    logging.warning(
        "nvidia-smi was not found; GPU temperature and utilization will be stored as NULL."
    )

# 2. Database Connection Handling
@st.cache_resource
def load_chroma_db():
    """Caches the database connection to prevent re-initialization on every UI interaction."""
    chroma_client = chromadb.PersistentClient(path="./movie_vector_db")
    
    embedding_function = OllamaEmbeddingFunction(
        url="http://localhost:11434",
        model_name="nomic-embed-text"
    )
    
    return chroma_client.get_collection(
        name="movies",
        embedding_function=embedding_function
    )

collection = load_chroma_db()

def retrieve_movies(query: str, limit: int = 10) -> str:
    """Executes vector search and surfaces both text AND hidden metadata."""
    results = collection.query(query_texts=[query], n_results=limit)
    
    if not results["documents"] or not results["documents"][0]:
        return "No relevant movies found in the database."
        
    formatted_chunks = []
    # Loop through the retrieved text chunks and their matching metadata dictionaries
    for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
        # Inject the metadata rating directly into the text block the LLM reads
        rating_text = f"Rating: {meta.get('rating', 'N/A')}/10"
        formatted_chunks.append(f"{doc} {rating_text}")
        
    return "\n\n".join(formatted_chunks)

def rewrite_query(user_input: str, chat_history: list) -> str:
    """Uses a fast LLM call to resolve pronouns before searching the database."""
    if not chat_history:
        return user_input # No history to resolve on the first turn
        
    # Format the last 4 messages to give the LLM context without overloading it
    history_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in chat_history[-4:]])
    
    rewrite_prompt = (
        "Given the conversation history, rewrite the user's latest message into a "
        "standalone search query. Replace vague pronouns like 'these', 'them', or 'it' "
        "with the actual movie titles being discussed. If it's already a standalone query, "
        "return it exactly as is. DO NOT answer the question, ONLY output the rewritten query.\n\n"
        f"History:\n{history_text}\n\n"
        f"Latest User Message: {user_input}\n\n"
        "Standalone Query:"
    )
    
    # Generate the rewritten query
    model_name = "llama3.1:8b"
    gpu_monitor = GpuMonitor()
    started_at = time.perf_counter()
    time_to_first_token_ms = None
    rewritten_query = ""
    final_chunk = None
    gpu_monitor.start()
    try:
        stream = ollama.chat(
            model=model_name,
            messages=[{"role": "user", "content": rewrite_prompt}],
            stream=True
        )
        for chunk in stream:
            final_chunk = chunk
            content = chunk["message"]["content"]
            if content and time_to_first_token_ms is None:
                time_to_first_token_ms = (time.perf_counter() - started_at) * 1000
            rewritten_query += content
    except Exception as exc:
        gpu_monitor.stop()
        log_model_call(
            model_name,
            "query_rewrite",
            started_at,
            final_chunk,
            gpu_monitor,
            time_to_first_token_ms,
            error=exc,
        )
        raise
    gpu_monitor.stop()
    log_model_call(
        model_name,
        "query_rewrite",
        started_at,
        final_chunk,
        gpu_monitor,
        time_to_first_token_ms,
    )

    return rewritten_query.strip()

# 3. Session State Initialization
if "messages" not in st.session_state:
    st.session_state.messages = []

# Render previous conversation history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 4. Chat Input & Processing Loop
if user_input := st.chat_input("Ask for a movie recommendation..."):
    # Display user message instantly
    with st.chat_message("user"):
        st.markdown(user_input)

    # Process and display agent response
    with st.chat_message("assistant"):
        with st.spinner("Analyzing context and searching local database..."):
            
            # Pass the history BEFORE appending the new user_input to avoid duplicating the prompt
            standalone_query = rewrite_query(user_input, st.session_state.messages)
            
            context = retrieve_movies(standalone_query)
            
        system_prompt = {
            "role": "system",
            "content": (
                "You are an expert movie recommendation AI. "
                "Use the provided database context as your primary source, but you may seamlessly draw on your "
                "internal knowledge to discuss movies, actors, and cinema history if the database lacks the answer. "
                "Do not apologize for using outside knowledge or announce that a movie, character, show, etc. isn't in your database. "
                "CRITICAL DOMAIN RESTRICTION: You must strictly refuse to answer any questions unrelated to movies, "
                "television, or cinema. If the user asks about cooking, sports, or other off-topic subjects, "
                "firmly but politely steer the conversation back to movies. "
                "Speak in a friendly and engaging manner, as if you are a movie enthusiast sharing recommendations with a friend."
                "Do not fabricate information. If you don't know the answer, say 'I don't know' or 'I cannot find a match in the database.'"
                "Do not give your own ratings. Only use the ratings provided in the database context or your internal knowledge. "
            )
        }
        
        # Inject RAG context into the latest query only
        augmented_prompt = (
            f"Context from database:\n<context>\n{context}\n</context>\n\n"
            f"User Question: {user_input}"
        )
        
        # Build memory array using past conversation state
        messages_payload = [system_prompt]
        for msg in st.session_state.messages[-6:]:
            messages_payload.append(msg)
        messages_payload.append({"role": "user", "content": augmented_prompt})

        response_placeholder = st.empty()
        full_response = ""
        
        # Stream from Ollama
        model_name = "llama3.1:8b"
        gpu_monitor = GpuMonitor()
        started_at = time.perf_counter()
        time_to_first_token_ms = None
        final_chunk = None
        gpu_monitor.start()
        try:
            stream = ollama.chat(
                model=model_name,
                messages=messages_payload,
                stream=True
            )

            for chunk in stream:
                final_chunk = chunk
                content = chunk['message']['content']
                if content and time_to_first_token_ms is None:
                    time_to_first_token_ms = (time.perf_counter() - started_at) * 1000
                full_response += content
                response_placeholder.markdown(full_response + "▌")
        except Exception as exc:
            gpu_monitor.stop()
            log_model_call(
                model_name,
                "answer_generation",
                started_at,
                final_chunk,
                gpu_monitor,
                time_to_first_token_ms,
                error=exc,
            )
            raise
        gpu_monitor.stop()
        log_model_call(
            model_name,
            "answer_generation",
            started_at,
            final_chunk,
            gpu_monitor,
            time_to_first_token_ms,
        )
            
        response_placeholder.markdown(full_response)
        
    # Append both the clean user input and the assistant response to memory AFTER processing completes
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.session_state.messages.append({"role": "assistant", "content": full_response})