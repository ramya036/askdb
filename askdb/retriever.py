"""FAISS-backed retrieval over schema descriptions and example question/SQL pairs.

Instead of stuffing the whole schema into every prompt, AskDB embeds each table
description and each worked example, then retrieves only the most relevant ones
for the user's question. This keeps prompts small and grounds the LLM in the
tables that actually matter.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import faiss
import numpy as np

EmbedFn = Callable[[Sequence[str]], np.ndarray]


def hashing_embedder(dim: int = 512) -> EmbedFn:
    """Dependency-free embedder (hashed word + bigram features). Used offline and in tests."""

    def embed(texts: Sequence[str]) -> np.ndarray:
        out = np.zeros((len(texts), dim), dtype="float32")
        for i, text in enumerate(texts):
            tokens = re.findall(r"[a-z0-9]+", text.lower())
            feats = tokens + [f"{a}_{b}" for a, b in zip(tokens, tokens[1:])]
            for f in feats:
                h = int(hashlib.md5(f.encode()).hexdigest(), 16)
                out[i, h % dim] += 1.0 if (h >> 1) % 2 else -1.0
        return out

    return embed


def gemini_embedder(api_key: str, model: str = "gemini-embedding-001") -> EmbedFn:
    from google import genai

    client = genai.Client(api_key=api_key)

    def embed(texts: Sequence[str]) -> np.ndarray:
        resp = client.models.embed_content(model=model, contents=list(texts))
        return np.array([e.values for e in resp.embeddings], dtype="float32")

    return embed


@dataclass
class Document:
    kind: str  # "table" or "example"
    text: str
    payload: dict


def _normalize(v: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return v / norms


class SchemaRetriever:
    def __init__(self, embed: EmbedFn):
        self.embed = embed
        self.docs: list[Document] = []
        self.index: faiss.Index | None = None

    def build(self, schema: list[dict], examples: list[dict], table_notes: dict[str, str] | None = None):
        table_notes = table_notes or {}
        self.docs = []
        for t in schema:
            cols = ", ".join(f"{c['name']} {c['type']}" for c in t["columns"])
            fks = "; ".join(f"{f['column']} -> {f['references']}" for f in t["foreign_keys"])
            note = table_notes.get(t["table"], "")
            text = f"Table {t['table']}: {note} Columns: {cols}." + (f" Foreign keys: {fks}." if fks else "")
            self.docs.append(Document("table", text, t))
        for ex in examples:
            self.docs.append(Document("example", f"Question: {ex['question']}", ex))

        vectors = _normalize(self.embed([d.text for d in self.docs]))
        self.index = faiss.IndexFlatIP(vectors.shape[1])  # cosine similarity on normalized vectors
        self.index.add(vectors)
        return self

    def search(self, question: str, k_tables: int = 4, k_examples: int = 3) -> dict:
        if self.index is None:
            raise RuntimeError("Retriever not built")
        q = _normalize(self.embed([question]))
        scores, ids = self.index.search(q, len(self.docs))
        tables, examples = [], []
        for score, idx in zip(scores[0], ids[0]):
            doc = self.docs[idx]
            if doc.kind == "table" and len(tables) < k_tables:
                tables.append({"text": doc.text, "table": doc.payload["table"], "score": float(score)})
            elif doc.kind == "example" and len(examples) < k_examples:
                examples.append({**doc.payload, "score": float(score)})
        return {"tables": tables, "examples": examples}


def load_examples(path: str | Path) -> list[dict]:
    return json.loads(Path(path).read_text())
