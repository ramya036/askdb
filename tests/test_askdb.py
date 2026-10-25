from pathlib import Path

import pytest

from askdb.app import TABLE_NOTES
from askdb.db import UnsafeQueryError, create_sample_db, describe_schema, run_readonly, validate_sql
from askdb.generator import NLToSQL, build_prompt, extract_sql
from askdb.retriever import SchemaRetriever, hashing_embedder, load_examples

EXAMPLES = Path(__file__).parent.parent / "askdb" / "data" / "examples.json"


@pytest.fixture(scope="module")
def db(tmp_path_factory):
    return create_sample_db(tmp_path_factory.mktemp("db") / "clinical.db", n_patients=50)


@pytest.fixture(scope="module")
def retriever(db):
    return SchemaRetriever(hashing_embedder()).build(describe_schema(db), load_examples(EXAMPLES), TABLE_NOTES)


def test_sample_db_is_populated(db):
    r = run_readonly(db, "SELECT COUNT(*) FROM patients")
    assert r["rows"][0][0] == 50
    assert run_readonly(db, "SELECT COUNT(*) FROM encounters")["rows"][0][0] >= 50


@pytest.mark.parametrize("sql", [
    "DELETE FROM patients",
    "SELECT 1; DROP TABLE patients",
    "UPDATE patients SET city = 'x'",
    "PRAGMA table_info(patients)",
    "",
])
def test_unsafe_sql_rejected(sql):
    with pytest.raises(UnsafeQueryError):
        validate_sql(sql)


def test_keywords_inside_strings_are_allowed():
    assert validate_sql("SELECT * FROM diagnoses WHERE description = 'drop foot';")


def test_readonly_connection_blocks_writes_even_if_validation_missed(db):
    import sqlite3
    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    with pytest.raises(sqlite3.OperationalError):
        conn.execute("DELETE FROM patients")
    conn.close()


def test_retriever_finds_relevant_tables(retriever):
    ctx = retriever.search("Which drugs are prescribed most often?")
    assert "medications" in [t["table"] for t in ctx["tables"]]
    assert ctx["examples"][0]["question"] == "Which medications are prescribed most often?"


def test_examples_sql_all_execute(db):
    for ex in load_examples(EXAMPLES):
        run_readonly(db, ex["sql"])


def test_extract_sql():
    assert extract_sql("Sure!\n```sql\nSELECT 1\n```") == "SELECT 1"
    assert extract_sql("SELECT 2") == "SELECT 2"


def test_pipeline_retries_after_sql_error(db, retriever):
    calls = []

    def fake_llm(prompt: str) -> str:
        calls.append(prompt)
        if len(calls) == 1:
            return "```sql\nSELECT nope FROM patients\n```"
        return "```sql\nSELECT state, COUNT(*) AS n FROM patients GROUP BY state\n```"

    out = NLToSQL(retriever, fake_llm, str(db)).ask("How many patients per state?")
    assert out["attempts"] == 2
    assert "Previous SQL" in calls[1]
    assert out["result"]["columns"] == ["state", "n"]


def test_pipeline_does_not_execute_unsafe_sql(db, retriever):
    out = NLToSQL(retriever, lambda p: "DROP TABLE patients", str(db)).ask("delete everything")
    assert "error" in out
    assert run_readonly(db, "SELECT COUNT(*) FROM patients")["rows"][0][0] == 50


def test_prompt_contains_context(retriever):
    ctx = retriever.search("average HbA1c of diabetic patients")
    prompt = build_prompt("average HbA1c of diabetic patients", ctx)
    assert "lab_results" in prompt and "Question: average HbA1c" in prompt
