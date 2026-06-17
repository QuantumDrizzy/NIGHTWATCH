import sqlite3
import datetime
import os

# Configurable so tests and alternate deployments don't touch the live store.
DB_FILE = os.environ.get("NIGHTWATCH_DB", "nightwatch.sqlite")

def init_db():
    """Initializes the SQLite database and creates / migrates the table."""
    conn   = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS contacts_log (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp       DATETIME DEFAULT CURRENT_TIMESTAMP,
            mco_class       TEXT    NOT NULL,
            azimuth         REAL    NOT NULL,
            altitude        REAL    NOT NULL,
            mahalanobis_d2  REAL,
            confidence      REAL,
            label           TEXT    DEFAULT '',
            omega           REAL    DEFAULT 0.0
        )
    ''')
    conn.commit()

    # Non-destructive migration: add columns if they were absent in an older schema
    for col, definition in [("label", "TEXT DEFAULT ''"),
                             ("omega", "REAL DEFAULT 0.0")]:
        try:
            cursor.execute(f"ALTER TABLE contacts_log ADD COLUMN {col} {definition}")
            conn.commit()
        except sqlite3.OperationalError:
            pass  # column already present

    conn.close()


def log_contact(mco_class: str, azimuth: float, altitude: float,
                mahalanobis_d2: float, confidence: float,
                label: str = "", omega: float = 0.0) -> None:
    """Insert one contact event into the blackbox."""
    conn   = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute(
        '''INSERT INTO contacts_log
               (timestamp, mco_class, azimuth, altitude,
                mahalanobis_d2, confidence, label, omega)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)''',
        (datetime.datetime.now().isoformat(sep=' '),
         mco_class, azimuth, altitude,
         mahalanobis_d2, confidence, label, omega)
    )
    conn.commit()
    conn.close()


def query_contacts(mco_class: str = None, limit: int = 5000):
    """Return rows as dicts, optionally filtered by class."""
    conn   = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    if mco_class:
        rows = conn.execute(
            "SELECT * FROM contacts_log WHERE mco_class = ? ORDER BY id DESC LIMIT ?",
            (mco_class, limit)
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM contacts_log ORDER BY id DESC LIMIT ?",
            (limit,)
        ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


# Initialize DB on module import
init_db()
