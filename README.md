# 🚀 AI Learning Assistant

AI Learning Assistant is a FastAPI-based backend application that generates personalized learning roadmaps, recommends portfolio projects, and answers roadmap-specific questions using Retrieval-Augmented Generation (RAG).

The application uses **Google Gemini**, **Gemini Embeddings**, **ChromaDB**, and **SQLite** to generate accurate, structured, and context-aware responses.

---

# Features

- AI-generated personalized learning roadmaps
- AI project recommendations
- Roadmap-aware chatbot using RAG
- Semantic search using ChromaDB
- Markdown roadmap export
- Structured JSON validation with Pydantic
- Docker support
- FastAPI with Swagger documentation

---

# Architecture Overview

```
User
   │
   ▼
FastAPI Backend
   │
   ├── Roadmap Service
   ├── Project Service
   └── Chat Service
          │
          ▼
 Google Gemini
          │
   Gemini Embeddings
          │
      ChromaDB
          │
   Retrieval-Augmented
        Responses
```

### Components

- **FastAPI** exposes REST APIs.
- **Gemini 2.5 Flash** generates structured roadmaps, projects, and chat responses.
- **Gemini Embeddings** generate semantic embeddings.
- **ChromaDB** stores roadmap embeddings for retrieval.
- **SQLite** stores roadmap metadata.
- **RAG Pipeline** retrieves relevant roadmap chunks before answering chat queries.

---

# Tech Stack

- Python 3.11
- FastAPI
- Google Gemini
- ChromaDB
- SQLAlchemy
- SQLite
- Docker
- Pydantic

---

# Setup Instructions

## Clone Repository

```bash
git clone https://github.com/mrityunjay5004/AI-Learning-Assistant.git

cd AI-Learning-Assistant
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

## Create `.env`

```env
GOOGLE_API_KEY=YOUR_API_KEY

GEMINI_MODEL=gemini-2.5-flash

GEMINI_EMBEDDING_MODEL=models/gemini-embedding-2

DATABASE_URL=sqlite:///./data/app.db

CHROMA_PERSIST_DIR=./data/chroma
```

## Run

```bash
uvicorn app.main:app --reload
```

Swagger

```
http://localhost:8000/docs
```

---

# Docker

Build

```bash
docker build -t ai-learning-assistant .
```

Run

```bash
docker run -p 8000:10000 \
-e GOOGLE_API_KEY=YOUR_API_KEY \
ai-learning-assistant
```

---

# API Endpoints

| Endpoint | Description |
|----------|-------------|
| GET /health | Health check |
| POST /roadmap | Generate roadmap |
| POST /project | Recommend project |
| POST /chat | Chat using roadmap context |
| GET /roadmap/{id}/markdown | Export roadmap |

---

# Assumptions Made

- Each roadmap belongs to one learning goal.
- Chat queries always reference an existing roadmap.
- Roadmaps are embedded immediately after generation.
- Chat responses are generated only from retrieved roadmap context.
- Single-user workflow is assumed.

---

# AI Tools / Frameworks Used

| Tool | Purpose |
|------|---------|
| Google Gemini 2.5 Flash | Roadmap, project and chat generation |
| Gemini Embeddings | Semantic embeddings |
| ChromaDB | Vector database |
| FastAPI | REST API |
| Pydantic | Schema validation |
| Tenacity | Retry logic |

---

# Prompt Design Decisions

- Separate system prompts and user prompts.
- Enforce structured JSON responses.
- Validate every response using Pydantic.
- Retry malformed responses automatically.
- Low temperature for deterministic outputs.
- Chat responses are grounded using retrieved roadmap context to reduce hallucinations.

---

# Deployment

The project is Dockerized and deployed on **Render**.

**Live API**

https://ai-learning-assistant-znmk.onrender.com

**Swagger Docs**

https://ai-learning-assistant-znmk.onrender.com/docs

---

# Demo Video

A 3–5 minute screen recording demonstrates:

- Roadmap generation
- Project recommendation
- Chat endpoint
- Swagger documentation
- Overall architecture

*(Add your Loom/Drive link before submission.)*

---

# Approximate Time Spent

Approximately **10 hours**, including:

- Backend development
- Gemini integration
- RAG implementation
- Testing & debugging
- Dockerization
- Deployment
- Documentation

---

# Author

**Mrityunjay Tiwari**

GitHub: https://github.com/mrityunjay5004

LinkedIn: https://www.linkedin.com/in/mrityunjaytiwari5004
