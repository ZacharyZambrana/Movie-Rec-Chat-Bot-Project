### Movie Recommendation Chat Bot Project Overview:
The goal of the project was to create a deeper understanding of RAG pipelines and architecture as well as learning to utilize open source models.
In this project I gathered IMDB movie and cast data from Kaggle where I joined and cleaned this data using python to load into ChromaDB.
Once in the vector database the model would be able to rank movies on similarity and probvide the user with exact answers from the DB to avoid halucination or misinformation.
The models are accessed through a streamlit UI that the user can prompt the model and start a new chat. I compared llama3.1:8B and Qwen3:8B models to where llama won by far.
I gathered model telemetry data to compare and contrast performance between models and created visualizations and tables in SQLite to showcase these findings.
