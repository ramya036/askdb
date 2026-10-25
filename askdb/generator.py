"""Prompt construction and SQL generation with a self-correction retry."""
from __future__ import annotations

import re
from typing import Callable, Protocol

from .db import UnsafeQueryError, run_readonly


class LLM(Protocol):
    def __call__(self, prompt: str) -> str: ...


def gemini_llm(api_key: str, model: str = "gemini-2.5-flash") -> LLM:
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=api_key)

    def call(prompt: str) -> str:
        resp = client.models.generate_content(
            model=model, contents=prompt, config=types.GenerateContentConfig(temperature=0.0)
        )
        return resp.text or ""

    return call


PROMPT = """You are an expert SQLite analyst for a clinical database.
Write ONE read-only SQLite query that answers the question.

Rules:
- Use only the tables and columns listed below.
- Return only SQL inside a ```sql code block. No explanation.
- Never modify data.
- Use explicit JOINs and readable column aliases.
- Today's date can be obtained with DATE('now').

Relevant tables:
{tables}

Similar solved examples:
{examples}
{error_section}
Question: {question}
"""


def build_prompt(question: str, context: dict, previous_sql: str | None = None, error: str | None = None) -> str:
    tables = "\n".join(f"- {t['text']}" for t in context["tables"])
    examples = "\n\n".join(f"Q: {e['question']}\nSQL: {e['sql']}" for e in context["examples"]) or "(none)"
    error_section = ""
    if previous_sql and error:
        error_section = (
            f"\nYour previous attempt failed.\nPrevious SQL: {previous_sql}\nError: {error}\n"
            "Fix the query.\n"
        )
    return PROMPT.format(tables=tables, examples=examples, error_section=error_section, question=question)


def extract_sql(text: str) -> str:
    match = re.search(r"```(?:sql)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    return (match.group(1) if match else text).strip()


class NLToSQL:
    def __init__(self, retriever, llm: LLM, db_path: str, max_attempts: int = 2):
        self.retriever = retriever
        self.llm = llm
        self.db_path = db_path
        self.max_attempts = max_attempts

    def ask(self, question: str, execute: bool = True) -> dict:
        context = self.retriever.search(question)
        sql, error = None, None
        for attempt in range(1, self.max_attempts + 1):
            sql = extract_sql(self.llm(build_prompt(question, context, sql, error)))
            if not execute:
                return {"question": question, "sql": sql, "attempts": attempt, "context": _ctx(context)}
            try:
                result = run_readonly(self.db_path, sql)
                return {"question": question, "sql": sql, "attempts": attempt, "result": result, "context": _ctx(context)}
            except UnsafeQueryError as e:
                # Never retry unsafe output: surface it.
                return {"question": question, "sql": sql, "attempts": attempt, "error": str(e), "context": _ctx(context)}
            except Exception as e:  # sqlite errors -> feed back to the model once
                error = str(e)
        return {"question": question, "sql": sql, "attempts": self.max_attempts, "error": error, "context": _ctx(context)}


def _ctx(context: dict) -> dict:
    return {"tables": [t["table"] for t in context["tables"]],
            "examples": [e["question"] for e in context["examples"]]}
