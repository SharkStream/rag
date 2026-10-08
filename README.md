# RAG Project

This project is a lightweight Retrieval-Augmented Generation (RAG) application built with Python. It includes:

- Flask as the API backend
- Gradio as the interactive UI
- Qdrant as the vector database for semantic retrieval
- OpenAI-compatible embedding and chat models for retrieval and generation
- Text chunking utilities for document preprocessing

## Project Structure

```text
RAG/
├── app.py                  # Flask application entry point
├── gradio_app.py          # Gradio frontend interface
├── config.py              # Configuration and environment settings
├── Dockerfile             # Docker image for running Qdrant
├── docker-compose.yml     # Docker Compose configuration for Qdrant
├── requirements.txt       # Python dependencies
├── README.md              # Project documentation
├── services/
│   ├── __init__.py
│   ├── llm_service.py     # Embedding and LLM integration
│   └── qdrant_service.py  # Qdrant interaction logic
└── utils/
    └── text_splitter.py   # Text chunking utilities
```

## Features

- Store and query vector embeddings in Qdrant
- Split large text into manageable chunks
- Search for relevant context using vector similarity
- Generate final responses with an LLM using retrieved context
- Run a simple web UI with Gradio for local testing

## Requirements

Before running the project, ensure you have the following installed:

- Python 3.10+
- Docker and Docker Compose (optional, for Qdrant)
- An OpenAI-compatible API key

## Quick Start

1. Copy the sample environment file:

```bash
copy .env.example .env
```

2. Update the values in the new `.env` file with your own API key and settings.

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Start Qdrant:

```bash
docker compose up -d
```

5. Run the app:

```bash
python app.py
```

6. Launch the Gradio interface:

```bash
python gradio_app.py
```

## Environment Variables

The project uses environment variables from a `.env` file. A sample template is included in [.env.example](.env.example).

```env
OPENAI_API_KEY=your_api_key_here
OPENAI_BASE_URL=https://api.openai.com/v1
EMBEDDING_MODEL=text-embedding-3-small
LLM_MODEL=gpt-4o-mini
QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=
QDRANT_COLLECTION=documents
QDRANT_VECTOR_SIZE=1536
DEFAULT_CHUNK_SIZE=500
DEFAULT_CHUNK_OVERLAP=50
DEBUG=false
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

## Run Qdrant with Docker

Using Docker Compose:

```bash
docker compose up -d
```

This starts Qdrant on:

- http://localhost:6333

## Run the Flask API

```bash
python app.py
```

The Flask app will run on:

```text
http://localhost:5000
```

## Run the Gradio Interface

```bash
python gradio_app.py
```

Then open:

```text
http://localhost:7860
```

## Usage Flow

1. Load or prepare a document
2. Split it into chunks using the text splitter
3. Generate embeddings with the LLM service
4. Store vectors in Qdrant
5. Query Qdrant with a user question
6. Retrieve relevant context
7. Send context + question to the LLM
8. Return the final answer to the user

## Notes

- The project is designed as a clean starter template for building a RAG app.
- The current configuration is intended for local development and testing.
- For production use, consider adding authentication, logging, retries, environment separation, and proper deployment settings.

## License

This project is for educational and development purposes.
