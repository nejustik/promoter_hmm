from collections.abc import Iterable
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
from typing import Any

from .bed import BedRecord


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS sequences (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    seq_id TEXT NOT NULL UNIQUE,
    description TEXT NOT NULL DEFAULT '',
    length INTEGER NOT NULL CHECK (length >= 0),
    gc_content REAL NOT NULL CHECK (gc_content >= 0.0 AND gc_content <= 1.0),
    source_path TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS models (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE,
    model_type TEXT NOT NULL,
    params_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_id INTEGER NOT NULL REFERENCES models(id) ON DELETE CASCADE,
    sequence_id INTEGER NOT NULL REFERENCES sequences(id) ON DELETE CASCADE,
    algorithm TEXT NOT NULL,
    log_likelihood REAL,
    metrics_json TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    chrom TEXT NOT NULL,
    start INTEGER NOT NULL CHECK (start >= 0),
    end INTEGER NOT NULL CHECK (end > start),
    name TEXT NOT NULL DEFAULT '.',
    score REAL NOT NULL DEFAULT 0.0,
    strand TEXT NOT NULL DEFAULT '.' CHECK (strand IN ('+', '-', '.'))
);

CREATE INDEX IF NOT EXISTS idx_runs_model_seq
    ON runs (model_id, sequence_id);

CREATE INDEX IF NOT EXISTS idx_predictions_run_id
    ON predictions (run_id);
"""


class Database:
    """
    Обертка над SQLite для хранения последовательностей, моделей и предсказаний
    """

    def __init__(self, db_path: str | Path = ":memory:") -> None:
        self.db_path = str(db_path)
        if self.db_path != ":memory:":
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON;")
        self.init_schema()

    def init_schema(self) -> None:
        with self.conn:
            self.conn.executescript(SCHEMA_SQL)

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> "Database":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    def upsert_sequence(
        self,
        seq_id: str,
        length: int,
        gc_content: float,
        description: str = "",
        source_path: str = "",
    ) -> int:
        now = _utc_now()
        with self.conn:
            self.conn.execute(
                """
                INSERT INTO sequences (seq_id, description, length, gc_content, source_path, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(seq_id) DO UPDATE SET
                    description = excluded.description,
                    length = excluded.length,
                    gc_content = excluded.gc_content,
                    source_path = excluded.source_path
                """,
                (seq_id, description, length, gc_content, source_path, now),
            )
            row = self.conn.execute(
                "SELECT id FROM sequences WHERE seq_id = ?",
                (seq_id,),
            ).fetchone()
        return int(row["id"])

    def get_sequence(self, seq_id: str) -> dict[str, Any] | None:
        row = self.conn.execute(
            "SELECT * FROM sequences WHERE seq_id = ?",
            (seq_id,),
        ).fetchone()
        return dict(row) if row is not None else None

    def save_model(
        self,
        name: str,
        model_type: str,
        params: dict[str, Any],
    ) -> int:
        now = _utc_now()
        params_json = json.dumps(params, ensure_ascii=False)
        with self.conn:
            self.conn.execute(
                """
                INSERT INTO models (name, model_type, params_json, created_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(name) DO UPDATE SET
                    model_type = excluded.model_type,
                    params_json = excluded.params_json,
                    created_at = excluded.created_at
                """,
                (name, model_type, params_json, now),
            )
            row = self.conn.execute(
                "SELECT id FROM models WHERE name = ?",
                (name,),
            ).fetchone()
        return int(row["id"])

    def load_model_params(self, name: str) -> dict[str, Any]:
        row = self.conn.execute(
            "SELECT params_json FROM models WHERE name = ?",
            (name,),
        ).fetchone()
        if row is None:
            raise KeyError(f"Модель {name!r} не найдена в базе данных")
        return json.loads(row["params_json"])

    def list_models(self) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT id, name, model_type, created_at FROM models ORDER BY id DESC"
        ).fetchall()
        return [dict(row) for row in rows]

    def create_run(
        self,
        model_id: int,
        sequence_id: int,
        algorithm: str,
        log_likelihood: float | None = None,
        metrics: dict[str, float] | None = None,
    ) -> int:
        now = _utc_now()
        metrics_json = json.dumps(metrics, ensure_ascii=False) if metrics is not None else None
        with self.conn:
            cursor = self.conn.execute(
                """
                INSERT INTO runs (model_id, sequence_id, algorithm, log_likelihood, metrics_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (model_id, sequence_id, algorithm, log_likelihood, metrics_json, now),
            )
        if cursor.lastrowid is None:
            raise RuntimeError("Не удалось получить ID созданного прогона")
        return cursor.lastrowid

    def save_predictions(self, run_id: int, records: Iterable[BedRecord]) -> int:
        rows = [
            (run_id, rec.chrom, rec.start, rec.end, rec.name, rec.score, rec.strand)
            for rec in records
        ]
        if not rows:
            return 0
        with self.conn:
            self.conn.executemany(
                """
                INSERT INTO predictions (run_id, chrom, start, end, name, score, strand)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                rows,
            )
        return len(rows)

    def get_predictions(self, run_id: int) -> list[BedRecord]:
        rows = self.conn.execute(
            """
            SELECT chrom, start, end, name, score, strand
            FROM predictions
            WHERE run_id = ?
            ORDER BY start ASC, end ASC
            """,
            (run_id,),
        ).fetchall()
        return [
            BedRecord(
                chrom=row["chrom"],
                start=row["start"],
                end=row["end"],
                name=row["name"],
                score=row["score"],
                strand=row["strand"],
            )
            for row in rows
        ]

    def get_runs(self, model_name: str | None = None) -> list[dict[str, Any]]:
        query = """
            SELECT
                r.id AS run_id,
                m.name AS model_name,
                m.model_type,
                s.seq_id,
                r.algorithm,
                r.log_likelihood,
                r.metrics_json,
                r.created_at
            FROM runs r
            JOIN models m ON m.id = r.model_id
            JOIN sequences s ON s.id = r.sequence_id
        """
        params: tuple[Any, ...] = ()
        if model_name is not None:
            query += " WHERE m.name = ?"
            params = (model_name,)
        query += " ORDER BY r.id DESC"

        rows = self.conn.execute(query, params).fetchall()
        result: list[dict[str, Any]] = []
        for row in rows:
            item = dict(row)
            item["metrics"] = (
                json.loads(item["metrics_json"]) if item["metrics_json"] is not None else None
            )
            del item["metrics_json"]
            result.append(item)
        return result