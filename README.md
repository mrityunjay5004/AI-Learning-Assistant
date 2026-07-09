# 🚀 AI Learning Assistant

AI Learning Assistant is a FastAPI-based backend application that generates personalized learning roadmaps, recommends portfolio projects, and answers roadmap-specific questions using Retrieval-Augmented Generation (RAG).

The application uses **Google Gemini**, **Gemini Embeddings**, **ChromaDB**, and **SQLite** to generate accurate, structured, and context-aware responses.

---

# 🌐 Live Demo

### 🚀 Live API

https://ai-learning-assistant-znmk.onrender.com

### 📖 Interactive API Documentation (Swagger UI)

https://ai-learning-assistant-znmk.onrender.com/docs

### 📄 OpenAPI Specification

https://ai-learning-assistant-znmk.onrender.com/openapi.json

### ❤️ Health Check

https://ai-learning-assistant-znmk.onrender.com/health

---

# 📦 Deployment

The application is fully containerized using Docker and deployed on **Render**.

### Live Application

- **API:** https://ai-learning-assistant-znmk.onrender.com
- **Swagger UI:** https://ai-learning-assistant-znmk.onrender.com/docs
- **OpenAPI JSON:** https://ai-learning-assistant-znmk.onrender.com/openapi.json
- **Health Check:** https://ai-learning-assistant-znmk.onrender.com/health

The project can also be deployed on any Docker-compatible cloud platform such as Render, Railway, or Fly.io.

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

# 🏗️ Architecture Overview

The AI Learning Assistant follows a modular, service-oriented architecture where each component has a single responsibility. The system combines LLM-based generation with Retrieval-Augmented Generation (RAG) to provide personalized and context-aware learning assistance.

```text
                           +----------------------+
                           |        Client        |
                           |  Swagger / Postman   |
                           +----------+-----------+
                                      |
                                      v
                          +-------------------------+
                          |     FastAPI Backend     |
                          |      REST Endpoints     |
                          +-----------+-------------+
                                      |
        +-----------------------------+-----------------------------+
        |                             |                             |
        v                             v                             v
+------------------+        +------------------+        +------------------+
| Roadmap Service  |        | Project Service  |        |   Chat Service   |
+--------+---------+        +--------+---------+        +--------+---------+
         |                           |                           |
         |                           |                           |
         +-------------+-------------+---------------------------+
                       |
                       v
              +----------------------+
              | Google Gemini 2.5    |
              | Flash LLM            |
              +----------+-----------+
                         |
          +--------------+--------------+
          |                             |
          v                             v
+----------------------+      +----------------------+
| Structured JSON      |      | Gemini Embeddings    |
| Roadmap / Project    |      +----------+-----------+
+----------------------+                 |
                                         v
                               +----------------------+
                               |      ChromaDB        |
                               | Vector Store (RAG)   |
                               +----------+-----------+
                                          |
                                          v
                               Retrieved Relevant Chunks
                                          |
                                          v
                               Context-Aware AI Response
```

## Component Responsibilities

### FastAPI
- Exposes REST APIs
- Handles request validation
- Returns structured JSON responses

### Roadmap Service
- Generates personalized learning roadmaps
- Estimates learning hours
- Organizes skills, tasks, subtasks, and resources

### Project Service
- Recommends portfolio projects based on the generated roadmap
- Suggests appropriate tech stacks and project features

### Chat Service (RAG)
- Converts user queries into embeddings
- Retrieves relevant roadmap chunks from ChromaDB
- Sends retrieved context to Gemini
- Returns grounded, roadmap-aware answers

### Google Gemini
- Generates structured roadmap and project recommendations
- Produces chat responses
- Generates semantic embeddings

### ChromaDB
- Stores roadmap embeddings
- Performs semantic similarity search
- Returns the most relevant chunks for Retrieval-Augmented Generation

### SQLite
- Stores roadmap metadata and application data
- Maintains roadmap IDs used during retrieval

---

## Request Flow

1. User sends a request to the FastAPI API.
2. Roadmap or project requests are processed directly using Google Gemini.
3. Generated roadmaps are chunked and embedded.
4. Embeddings are stored in ChromaDB.
5. During chat, the user query is embedded.
6. ChromaDB retrieves the most relevant roadmap chunks.
7. Retrieved context is combined with the user query.
8. Gemini generates a grounded response using the retrieved context.
9. FastAPI returns the final structured response.

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

*(Adding.)*

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
