"""Offline validation harness for the NIGHTWATCH detection store (nightwatch_db).

Pure stdlib sqlite — no torch/CUDA/sensor needed. Uses a temporary DB so the live
RAG store is never touched.
"""

from __future__ import annotations

import os
import sqlite3
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import nightwatch_db  # noqa: E402


@pytest.fixture
def temp_db(tmp_path, monkeypatch):
    db = tmp_path / "test.sqlite"
    monkeypatch.setattr(nightwatch_db, "DB_FILE", str(db))
    nightwatch_db.init_db()
    return str(db)


EXPECTED_COLUMNS = {
    "id", "timestamp", "mco_class", "azimuth", "altitude",
    "mahalanobis_d2", "confidence", "label", "omega",
}


def test_schema_has_all_columns(temp_db):
    conn = sqlite3.connect(temp_db)
    cols = {r[1] for r in conn.execute("PRAGMA table_info(contacts_log)").fetchall()}
    conn.close()
    assert EXPECTED_COLUMNS <= cols


def test_log_and_query_roundtrip(temp_db):
    nightwatch_db.log_contact("debris", 123.4, 45.6, 2.1, 0.9, label="trk-1", omega=0.05)
    rows = nightwatch_db.query_contacts()
    assert len(rows) == 1
    r = rows[0]
    assert r["mco_class"] == "debris"
    assert abs(r["azimuth"] - 123.4) < 1e-6
    assert abs(r["altitude"] - 45.6) < 1e-6
    assert r["label"] == "trk-1"
    assert abs(r["omega"] - 0.05) < 1e-9


def test_filter_by_class(temp_db):
    nightwatch_db.log_contact("aircraft", 1, 1, 0, 0)
    nightwatch_db.log_contact("debris", 2, 2, 0, 0)
    nightwatch_db.log_contact("aircraft", 3, 3, 0, 0)
    assert len(nightwatch_db.query_contacts("aircraft")) == 2
    assert len(nightwatch_db.query_contacts("debris")) == 1
    assert len(nightwatch_db.query_contacts()) == 3


def test_query_orders_most_recent_first(temp_db):
    nightwatch_db.log_contact("x", 1, 1, 0, 0)
    nightwatch_db.log_contact("x", 2, 2, 0, 0)
    rows = nightwatch_db.query_contacts()
    assert rows[0]["id"] > rows[1]["id"]


def test_limit_respected(temp_db):
    for i in range(10):
        nightwatch_db.log_contact("x", i, i, 0, 0)
    assert len(nightwatch_db.query_contacts(limit=3)) == 3


def test_init_db_idempotent_and_nondestructive(temp_db):
    nightwatch_db.log_contact("x", 0, 0, 0, 0)
    nightwatch_db.init_db()                     # re-init must not wipe or error
    nightwatch_db.init_db()                     # migration path on a populated table
    assert len(nightwatch_db.query_contacts()) == 1


def test_optional_fields_default(temp_db):
    nightwatch_db.log_contact("anomalous", 10.0, 20.0, 1.5, 0.7)   # no label/omega
    r = nightwatch_db.query_contacts()[0]
    assert r["label"] == ""
    assert abs(r["omega"]) < 1e-12
