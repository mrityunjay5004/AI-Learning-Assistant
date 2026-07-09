# 🚀 AI Learning Assistant

![Python](https://img.shields.io/badge/Python-3.11-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-green)
![Gemini](https://img.shields.io/badge/Google-Gemini-orange)
![ChromaDB](https://img.shields.io/badge/Vector%20DB-Chroma-purple)
![Docker](https://img.shields.io/badge/Docker-Supported-2496ED?logo=docker&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-blue)

An AI-powered learning assistant that generates personalized learning roadmaps, recommends hands-on projects, and answers roadmap-specific questions using Retrieval-Augmented Generation (RAG).

Built with **FastAPI**, **Google Gemini**, **ChromaDB**, **SQLite**, and **Docker**.

---

# 🌐 Live Demo

### API

https://ai-learning-assistant-znmk.onrender.com

### Interactive API Documentation

https://ai-learning-assistant-znmk.onrender.com/docs

---

# ✨ Features

- 🎯 Personalized AI-generated learning roadmaps
- 📚 Skill-based milestone planning
- 💻 AI-generated portfolio project recommendations
- 🤖 Roadmap-aware chatbot using RAG
- 🔍 Semantic search with Chroma Vector Database
- 📝 Markdown roadmap export
- ✅ Strict JSON validation using Pydantic
- 🐳 Dockerized for easy deployment
- ⚡ FastAPI with automatic OpenAPI documentation

---

# 🏗️ Architecture

```
                   User
                     │
                     ▼
              FastAPI Backend
                     │
      ┌──────────────┼──────────────┐
      │              │              │
      ▼              ▼              ▼
 Gemini LLM     SQLite DB     Chroma Vector DB
      │                              │
      └──────────────┬───────────────┘
                     ▼
            Retrieval-Augmented Chat
```

---

# 🛠️ Tech Stack

## Backend

- FastAPI
- Python 3.11
- SQLAlchemy
- SQLite

## AI

- Google Gemini 2.5 Flash
- Gemini Embeddings
- Prompt Engineering

## RAG

- ChromaDB
- Semantic Search
- Vector Embeddings

## Validation

- Pydantic v2

## Deployment

- Docker
- Render

---

# 📂 Project Structure

```
app/
│
├── db/
├── models/
├── rag/
├── routers/
├── services/
├── prompts.py
├── resources.py
├── config.py
├── exceptions.py
└── main.py

tests/

requirements.txt
Dockerfile
README.md
```

---

# ⚙️ Installation

## Clone Repository

```bash
git clone https://github.com/mrityunjay5004/AI-Learning-Assistant.git

cd AI-Learning-Assistant
```

---

## Create Virtual Environment

### Windows

```bash
python -m venv venv

venv\Scripts\activate
```

### Linux / macOS

```bash
python3 -m venv venv

source venv/bin/activate
```

---

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

# 🔑 Environment Variables

Create a `.env` file.

```env
GOOGLE_API_KEY=YOUR_GEMINI_API_KEY

GEMINI_MODEL=gemini-2.5-flash

GEMINI_EMBEDDING_MODEL=models/gemini-embedding-2

DATABASE_URL=sqlite:///./data/app.db

CHROMA_PERSIST_DIR=./data/chroma

LOG_LEVEL=INFO

LLM_MAX_RETRIES=3

RAG_TOP_K=4
```

---

# ▶️ Run Locally

```bash
uvicorn app.main:app --reload
```

Visit

```
http://localhost:8000/docs
```

---

# 🐳 Docker

## Build Image

```bash
docker build -t ai-learning-assistant .
```

## Run Container

```bash
docker run \
-p 8000:10000 \
-e GOOGLE_API_KEY=YOUR_API_KEY \
ai-learning-assistant
```

Swagger Documentation

```
http://localhost:8000/docs
```

---

# 🚀 API Endpoints

## Health

```
GET /health
```

Checks whether the API is running.

---

## Generate Learning Roadmap

```
POST /roadmap
```

Example

```json
{
  "goal_title": "Backend Developer",
  "experience": "Less than 1 year",
  "known_skills": [
    "Python",
    "SQL"
  ],
  "learning_style": "Project Based",
  "weekly_hours": 15
}
```

---

## Generate Project Recommendation

```
POST /project
```

Example

```json
{
  "roadmap_id": "<ROADMAP_ID>"
}
```

---

## Chat with Roadmap

```
POST /chat
```

Example

```json
{
  "roadmap_id": "<ROADMAP_ID>",
  "message": "What should I learn first?"
}
```

---

## Export Markdown

```
GET /roadmap/{roadmap_id}/markdown
```

Exports the roadmap as Markdown.

---

# 🧠 Retrieval-Augmented Generation (RAG)

The chatbot answers questions using the generated roadmap instead of relying solely on the LLM.

Workflow:

1. Roadmap is generated.
2. Roadmap is chunked.
3. Chunks are embedded using Gemini Embeddings.
4. Stored inside ChromaDB.
5. User question is embedded.
6. Relevant chunks retrieved.
7. Context + Question sent to Gemini.
8. Grounded response returned.

---

# 🧪 Running Tests

```bash
pytest
```

---

# 📦 Deployment

This project is containerized with Docker and can be deployed on:

- Render
- Railway
- Fly.io
- Any Docker-compatible cloud platform

The deployed application includes automatic Swagger documentation.

---

# 📈 Future Improvements

- User authentication
- Learning progress tracking
- Roadmap editing
- Multiple LLM providers
- Course recommendations
- PDF roadmap export
- Learning analytics dashboard
- Multi-language support

---

# 👨‍💻 Author

**Mrityunjay Tiwari**

GitHub

https://github.com/mrityunjay5004

LinkedIn

https://www.linkedin.com/in/mrityunjaytiwari5004/

---

# ⭐ Support

If you found this project useful, consider giving it a ⭐ on GitHub.
