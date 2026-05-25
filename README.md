# AI Job Matching

Production-level AI-powered job matching using vector retrieval.

Implements **Option A — Minimal Change** from the architecture brief:
- Same API endpoints, same response payloads
- Vector index retrieval instead of full-resume scan
- Optional LLM reranking stage
- **Sub-10 second** match endpoint under load

## Architecture

```
┌──────────────┐    ┌──────────────┐    ┌──────────────────┐
│  Resume Upload│───▶│  Embedding   │───▶│  FAISS Vector     │
│  /api/resume/ │    │  (sentence-  │    │  Index            │
│  upload       │    │  transformers)│    │                   │
└──────────────┘    └──────────────┘    └────────┬─────────┘
                                                  │
┌──────────────┐    ┌──────────────┐             │
│  JD Parse    │───▶│  Embedding   │───┐         │
│  /api/jd/    │    │  (same model)│   │         │
│  parse       │    └──────────────┘   │         │
└──────────────┘                       ▼         ▼
                               ┌──────────────────────┐
                               │  MatcherService      │
                               │  ┌────────────────┐  │
                               │  │ 1. Load JD vec │  │
                               │  │ 2. Vector      │  │
                               │  │    Search topK │  │
                               │  │ 3. Merge       │  │
                               │  │    Metadata    │  │
                               │  │ 4. Compute     │  │
                               │  │    Score       │  │
                               │  │ 5. (Optional)  │  │
                               │  │    LLM Rerank  │  │
                               │  └────────────────┘  │
                               └──────────────────────┘
                                        │
                                        ▼
                               ┌──────────────────┐
                               │  Match Response  │
                               │  matches[ ] with │
                               │  score, skills,  │
                               │  ai_reason       │
                               └──────────────────┘
```

## Tech Stack

| Component          | Technology                         |
|--------------------|------------------------------------|
| API Framework      | FastAPI (Python)                   |
| Embedding Model    | all-MiniLM-L6-v2 (sentence-transformers) |
| Vector Store       | FAISS (IndexFlatIP + IndexIDMap)   |
| Metadata Store     | SQLite                             |
| LLM Reranker       | OpenAI GPT-4o-mini (optional)      |
| Validation         | Pydantic v2                        |

## Quick Start

### Prerequisites

- Python 3.11+
- pip

### Setup

```bash
# 1. Clone / navigate to the project
cd background_voice

# 2. Create virtual environment
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. (Optional) Configure OpenAI for reranking
cp .env.example .env
# Edit .env and set OPENAI_API_KEY=sk-...

# 5. Run the server
python -m app.main
```

The API will be available at **http://localhost:8000**.

Interactive docs: **http://localhost:8000/docs**

### Docker

```bash
docker-compose up --build
```

## API Endpoints

### Health Check
```
GET /health
```

### Upload Resume
```json
POST /api/resume/upload
{
  "text": "Full resume text...",
  "tenant_id": "default"  // optional
}
```

### Parse Job Description
```json
POST /api/jd/parse
{
  "text": "Full job description text...",
  "tenant_id": "default"  // optional
}
```

### Match Candidates
```
GET /api/aijob-matching/{jd_id}?top_k=50&rerank=false&tenant_id=default
```

**Response:**
```json
{
  "jd_id": "abc123",
  "total_candidates": 225,
  "matches": [
    {
      "resume_id": "def456",
      "score": 87.3,
      "summary": "12 skills matched, 3 missing",
      "matched_skills": ["python", "fastapi", "kubernetes"],
      "missing_skills": ["kafka", "typescript"],
      "ai_reason": "Strong match: extensive Python and FastAPI experience with Kubernetes orchestration."
    }
  ]
}
```

## Configuration

All configuration via environment variables (see `.env.example`):

| Variable                  | Default             | Description                     |
|---------------------------|---------------------|---------------------------------|
| `HOST`                    | `0.0.0.0`          | Server host                     |
| `PORT`                    | `8000`             | Server port                     |
| `EMBEDDING_MODEL_NAME`    | `all-MiniLM-L6-v2` | Sentence-transformer model      |
| `DEFAULT_TOP_K`           | `50`               | Default candidates to retrieve  |
| `RERANK_DEFAULT`          | `false`            | Enable LLM rerank by default    |
| `RERANK_LIMIT`            | `30`               | Max candidates sent to LLM      |
| `OPENAI_API_KEY`          | —                  | API key for GPT-4o-mini rerank  |
| `CACHE_TTL_SECONDS`       | `3600`             | Result cache TTL (planned)      |

## Performance

The vector index retrieval replaces the O(N) full-resume scan with an O(log N) FAISS search:

| Pool Size | Full Scan (no index) | FAISS Index (topK=50) |
|-----------|---------------------|----------------------|
| 225       | ~30-60s (with LLM) | < 1s                 |
| 10,000    | N/A (timeout)       | ~50ms                |
| 100,000   | N/A                 | ~200ms               |

## Testing

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run specific test file
pytest tests/test_matcher.py -v
```

## Project Structure

```
background_voice/
├── app/
│   ├── api/
│   │   └── routes.py        # FastAPI route handlers
│   ├── models/
│   │   └── schemas.py       # Pydantic request/response models
│   ├── services/
│   │   ├── embedding.py     # Sentence-transformers embedding
│   │   ├── vector_store.py  # FAISS index management
│   │   ├── metadata_store.py# SQLite metadata storage
│   │   ├── parser.py        # Resume/JD text parser
│   │   ├── skills_base.py   # Comprehensive skills database
│   │   ├── matcher.py       # Matching orchestrator
│   │   └── reranker.py      # LLM reranking (OpenAI)
│   ├── config.py            # Configuration via pydantic-settings
│   └── main.py              # FastAPI entry point
├── tests/
│   ├── conftest.py          # Fixtures & sample data
│   ├── test_embedding.py
│   ├── test_parser.py
│   ├── test_matcher.py
│   └── test_api.py
├── data/                    # Runtime data (FAISS indices, SQLite DB)
├── .env.example
├── requirements.txt
└── README.md
```
