# AskDB — Natural Language to SQL

Ask questions about a database in plain English and get back the SQL **and** the results.
AskDB uses **retrieval-augmented generation (RAG)**: it retrieves only the relevant tables and similar solved examples from a **FAISS** vector index, then asks **Gemini** to write a single read-only SQLite query.

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white)
![FAISS](https://img.shields.io/badge/FAISS-vector_search-0467DF)
![Gemini](https://img.shields.io/badge/Gemini_API-8E75B2?logo=googlegemini&logoColor=white)

```
Question ─▶ Embed ─▶ FAISS search ─▶ top tables + similar examples
                                          │
                                          ▼
                              Gemini prompt ─▶ SQL
                                          │
                     validate (SELECT only) ─▶ run on read-only DB
                                          │
                       SQL error? ─▶ feed error back, retry once
                                          ▼
                                   columns + rows
```

## Features

- **RAG over the schema:** table descriptions and example question/SQL pairs are embedded and indexed with FAISS, so prompts stay small even for large schemas.
- **Few-shot grounding:** the most similar solved examples are injected into the prompt.
- **Self-correction:** if SQLite rejects the query, the error is sent back to the model for one fix attempt.
- **Safe by design:** two layers of protection. Generated SQL must be a single `SELECT`/`WITH` statement, and it runs on a `mode=ro` SQLite connection that cannot write even if validation were bypassed.
- **Synthetic clinical dataset:** patients, providers, encounters, ICD-10 diagnoses, medications and lab results. No real patient data.
- **Web UI, REST API and CLI.**

## Quick start

```bash
git clone https://github.com/ramya036/askdb.git
cd askdb
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env    # then add your GEMINI_API_KEY (free at https://aistudio.google.com/apikey)
uvicorn askdb.app:app --reload
```

Open http://localhost:8000 and try *"Which medications are prescribed most for heart failure?"*

The sample database is created automatically on first run at `data/clinical.db`.

### Docker

```bash
docker build -t askdb .
docker run -p 8000:8000 -e GEMINI_API_KEY=your_key askdb
```

### CLI

```bash
python -m askdb.cli "Average HbA1c for diabetic patients by gender"
```

## API

`POST /api/ask`

```json
{ "question": "How many patients live in each state?" }
```

Response:

```json
{
  "sql": "SELECT state, COUNT(*) AS patient_count FROM patients GROUP BY state ORDER BY patient_count DESC",
  "attempts": 1,
  "result": { "columns": ["state", "patient_count"], "rows": [["GA", 81], ["TX", 42]], "truncated": false },
  "context": { "tables": ["patients", "encounters", "providers", "diagnoses"], "examples": ["How many patients are there in each state?"] }
}
```

Other endpoints: `GET /api/schema`, `GET /api/health`. Interactive docs at `/docs`.

## Project structure

```
askdb/
├── app.py          FastAPI app and pipeline wiring
├── retriever.py    Embeddings + FAISS index over schema and examples
├── generator.py    Prompt building, Gemini call, retry on SQL errors
├── db.py           Synthetic clinical DB, SQL validation, read-only execution
├── cli.py          Command-line interface
├── data/examples.json   Few-shot question/SQL pairs
└── static/index.html    Web UI
tests/              pytest suite (runs offline, no API key needed)
```

## Configuration

| Variable | Default | Description |
|---|---|---|
| `GEMINI_API_KEY` | — | Required for SQL generation |
| `ASKDB_EMBEDDER` | `gemini` | `gemini` for Gemini embeddings, `hashing` for an offline embedder |
| `ASKDB_DB_PATH` | `data/clinical.db` | Point at your own SQLite database |

To use your own database, set `ASKDB_DB_PATH`, add table notes in `app.py` and replace `data/examples.json` with examples from your domain.

## Tests

```bash
pip install -r requirements-dev.txt
pytest
```

Tests use an offline embedder and a fake LLM, so they need no network or API key.

## Roadmap

- PostgreSQL support
- Query result charts
- Evaluation harness measuring execution accuracy on a labeled question set
