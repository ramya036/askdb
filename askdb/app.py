"""FastAPI service: POST /api/ask turns a plain-English question into SQL and runs it."""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from .db import create_sample_db, describe_schema
from .generator import NLToSQL, gemini_llm
from .retriever import SchemaRetriever, gemini_embedder, hashing_embedder, load_examples

load_dotenv()

PKG = Path(__file__).parent
DB_PATH = Path(os.getenv("ASKDB_DB_PATH", PKG.parent / "data" / "clinical.db"))

TABLE_NOTES = {
    "patients": "Demographics for each patient: name, birth date, gender, city and state.",
    "providers": "Doctors and their medical specialty.",
    "encounters": "Each hospital or clinic visit: date, type (outpatient, inpatient, emergency), length of stay.",
    "diagnoses": "ICD-10 diagnosis codes and descriptions recorded during an encounter, e.g. diabetes, hypertension, asthma.",
    "medications": "Drugs prescribed during an encounter with dose and days supply.",
    "lab_results": "Lab test values from an encounter such as HbA1c, LDL, BNP, blood pressure.",
}


class AskRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=500)
    execute: bool = True


@lru_cache
def get_pipeline() -> NLToSQL:
    if not DB_PATH.exists():
        create_sample_db(DB_PATH)
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("Set GEMINI_API_KEY to generate SQL (see .env.example)")
    embed = gemini_embedder(api_key) if os.getenv("ASKDB_EMBEDDER", "gemini") == "gemini" else hashing_embedder()
    retriever = SchemaRetriever(embed).build(
        describe_schema(DB_PATH), load_examples(PKG / "data" / "examples.json"), TABLE_NOTES
    )
    return NLToSQL(retriever, gemini_llm(api_key), str(DB_PATH))


app = FastAPI(title="AskDB", description="Natural language to SQL with RAG + FAISS + Gemini")


@app.get("/")
def index():
    return FileResponse(PKG / "static" / "index.html")


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/schema")
def schema():
    if not DB_PATH.exists():
        create_sample_db(DB_PATH)
    return describe_schema(DB_PATH)


@app.post("/api/ask")
def ask(req: AskRequest):
    try:
        pipeline = get_pipeline()
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    return pipeline.ask(req.question, execute=req.execute)
