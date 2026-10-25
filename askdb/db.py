"""Sample clinical database and safe, read-only query execution."""
from __future__ import annotations

import random
import re
import sqlite3
from datetime import date, timedelta
from pathlib import Path

SCHEMA = """
CREATE TABLE patients (
    patient_id   INTEGER PRIMARY KEY,
    first_name   TEXT NOT NULL,
    last_name    TEXT NOT NULL,
    birth_date   DATE NOT NULL,
    gender       TEXT CHECK (gender IN ('F', 'M', 'O')),
    city         TEXT,
    state        TEXT
);

CREATE TABLE providers (
    provider_id  INTEGER PRIMARY KEY,
    full_name    TEXT NOT NULL,
    specialty    TEXT NOT NULL
);

CREATE TABLE encounters (
    encounter_id   INTEGER PRIMARY KEY,
    patient_id     INTEGER NOT NULL REFERENCES patients(patient_id),
    provider_id    INTEGER NOT NULL REFERENCES providers(provider_id),
    encounter_date DATE NOT NULL,
    encounter_type TEXT CHECK (encounter_type IN ('outpatient', 'inpatient', 'emergency')),
    length_of_stay_days INTEGER DEFAULT 0
);

CREATE TABLE diagnoses (
    diagnosis_id   INTEGER PRIMARY KEY,
    encounter_id   INTEGER NOT NULL REFERENCES encounters(encounter_id),
    icd10_code     TEXT NOT NULL,
    description    TEXT NOT NULL
);

CREATE TABLE medications (
    medication_id  INTEGER PRIMARY KEY,
    encounter_id   INTEGER NOT NULL REFERENCES encounters(encounter_id),
    drug_name      TEXT NOT NULL,
    dose_mg        REAL,
    days_supply    INTEGER
);

CREATE TABLE lab_results (
    lab_id         INTEGER PRIMARY KEY,
    encounter_id   INTEGER NOT NULL REFERENCES encounters(encounter_id),
    test_name      TEXT NOT NULL,
    value          REAL NOT NULL,
    unit           TEXT NOT NULL
);
"""

_FIRST = ["Ava", "Liam", "Maya", "Noah", "Priya", "Ethan", "Zoe", "Arjun", "Mia", "Lucas", "Sara", "Omar"]
_LAST = ["Patel", "Smith", "Garcia", "Chen", "Johnson", "Reddy", "Brown", "Nguyen", "Lee", "Davis"]
_CITIES = [("Atlanta", "GA"), ("Savannah", "GA"), ("Austin", "TX"), ("Seattle", "WA"), ("Boston", "MA")]
_PROVIDERS = [
    ("Dr. Anna Ruiz", "Cardiology"), ("Dr. Ben Okafor", "Endocrinology"),
    ("Dr. Chloe Kim", "Primary Care"), ("Dr. David Shah", "Emergency Medicine"),
    ("Dr. Elena Rossi", "Pulmonology"),
]
_DIAGNOSES = [
    ("E11.9", "Type 2 diabetes mellitus without complications", "Endocrinology", "HbA1c", "%", (6.5, 10.5)),
    ("I10", "Essential (primary) hypertension", "Cardiology", "Systolic BP", "mmHg", (130, 175)),
    ("J45.909", "Asthma, unspecified", "Pulmonology", "FEV1", "L", (1.5, 3.5)),
    ("I50.9", "Heart failure, unspecified", "Cardiology", "BNP", "pg/mL", (100, 900)),
    ("J18.9", "Pneumonia, unspecified organism", "Emergency Medicine", "WBC", "10^3/uL", (11, 20)),
    ("E78.5", "Hyperlipidemia, unspecified", "Primary Care", "LDL", "mg/dL", (130, 220)),
]
_MEDS = {
    "E11.9": [("Metformin", 500), ("Insulin glargine", 20)],
    "I10": [("Lisinopril", 10), ("Amlodipine", 5)],
    "J45.909": [("Albuterol", 2.5)],
    "I50.9": [("Furosemide", 40), ("Carvedilol", 12.5)],
    "J18.9": [("Azithromycin", 250)],
    "E78.5": [("Atorvastatin", 20)],
}


def create_sample_db(path: str | Path, n_patients: int = 200, seed: int = 7) -> Path:
    """Create a reproducible synthetic clinical database. No real patient data."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    rng = random.Random(seed)
    conn = sqlite3.connect(path)
    conn.executescript(SCHEMA)
    conn.executemany("INSERT INTO providers VALUES (?, ?, ?)",
                     [(i + 1, n, s) for i, (n, s) in enumerate(_PROVIDERS)])
    provider_by_specialty = {s: i + 1 for i, (_, s) in enumerate(_PROVIDERS)}

    enc_id = diag_id = med_id = lab_id = 0
    start = date(2023, 1, 1)
    for pid in range(1, n_patients + 1):
        city, state = rng.choice(_CITIES)
        birth = date(1940, 1, 1) + timedelta(days=rng.randint(0, 365 * 65))
        conn.execute("INSERT INTO patients VALUES (?, ?, ?, ?, ?, ?, ?)",
                     (pid, rng.choice(_FIRST), rng.choice(_LAST), birth.isoformat(),
                      rng.choice(["F", "M"]), city, state))
        for _ in range(rng.randint(1, 4)):
            enc_id += 1
            code, desc, specialty, test, unit, (lo, hi) = rng.choice(_DIAGNOSES)
            etype = rng.choices(["outpatient", "inpatient", "emergency"], [6, 2, 2])[0]
            los = rng.randint(1, 9) if etype == "inpatient" else 0
            when = start + timedelta(days=rng.randint(0, 700))
            conn.execute("INSERT INTO encounters VALUES (?, ?, ?, ?, ?, ?)",
                         (enc_id, pid, provider_by_specialty[specialty], when.isoformat(), etype, los))
            diag_id += 1
            conn.execute("INSERT INTO diagnoses VALUES (?, ?, ?, ?)", (diag_id, enc_id, code, desc))
            for drug, dose in _MEDS[code]:
                if rng.random() < 0.8:
                    med_id += 1
                    conn.execute("INSERT INTO medications VALUES (?, ?, ?, ?, ?)",
                                 (med_id, enc_id, drug, dose, rng.choice([7, 30, 90])))
            lab_id += 1
            conn.execute("INSERT INTO lab_results VALUES (?, ?, ?, ?, ?)",
                         (lab_id, enc_id, test, round(rng.uniform(lo, hi), 1), unit))
    conn.commit()
    conn.close()
    return path


_FORBIDDEN = re.compile(
    r"\b(insert|update|delete|drop|alter|create|replace|attach|detach|pragma|vacuum|reindex|truncate)\b",
    re.IGNORECASE,
)


class UnsafeQueryError(ValueError):
    """Raised when generated SQL is not a single read-only statement."""


def validate_sql(sql: str) -> str:
    """Allow exactly one SELECT/WITH statement. Defence in depth on top of a read-only connection."""
    cleaned = sql.strip().rstrip(";").strip()
    if not cleaned:
        raise UnsafeQueryError("Empty query")
    if ";" in cleaned:
        raise UnsafeQueryError("Only one statement is allowed")
    if not re.match(r"^(select|with)\b", cleaned, re.IGNORECASE):
        raise UnsafeQueryError("Only SELECT queries are allowed")
    if _FORBIDDEN.search(re.sub(r"'[^']*'", "''", cleaned)):
        raise UnsafeQueryError("Query contains a write or admin keyword")
    return cleaned


def run_readonly(db_path: str | Path, sql: str, max_rows: int = 200) -> dict:
    """Execute validated SQL on a read-only connection and return columns + rows."""
    sql = validate_sql(sql)
    uri = f"file:{Path(db_path).resolve()}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    try:
        cur = conn.execute(sql)
        columns = [c[0] for c in cur.description or []]
        rows = cur.fetchmany(max_rows + 1)
    finally:
        conn.close()
    return {
        "columns": columns,
        "rows": [list(r) for r in rows[:max_rows]],
        "truncated": len(rows) > max_rows,
    }


def describe_schema(db_path: str | Path) -> list[dict]:
    """Return one entry per table with its columns and foreign keys, read from the database itself."""
    uri = f"file:{Path(db_path).resolve()}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    try:
        tables = [r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name")]
        out = []
        for t in tables:
            cols = [{"name": c[1], "type": c[2]} for c in conn.execute(f"PRAGMA table_info({t})")]
            fks = [{"column": f[3], "references": f"{f[2]}.{f[4]}"} for f in conn.execute(f"PRAGMA foreign_key_list({t})")]
            out.append({"table": t, "columns": cols, "foreign_keys": fks})
        return out
    finally:
        conn.close()
