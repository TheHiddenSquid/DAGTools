"""Local SQLite metadata and immutable NumPy datasets."""
import csv
import hashlib
import io
import itertools
import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def now():
    return datetime.now(timezone.utc).isoformat()


class Store:
    def __init__(self, root: Path):
        self.root = root.resolve()
        (self.root / "datasets").mkdir(parents=True, exist_ok=True)
        (self.root / "runs").mkdir(exist_ok=True)
        self.db = self.root / "experiments.sqlite3"
        with self.connect() as db:
            db.execute("CREATE TABLE IF NOT EXISTS records (kind TEXT, id TEXT, payload TEXT, PRIMARY KEY(kind,id))")

    def connect(self):
        return sqlite3.connect(self.db, timeout=20)

    def put(self, kind, record):
        with self.connect() as db:
            db.execute("INSERT OR REPLACE INTO records VALUES (?,?,?)", (kind, record["id"], json.dumps(record, allow_nan=False)))
        return record

    def get(self, kind, key):
        with self.connect() as db:
            row = db.execute("SELECT payload FROM records WHERE kind=? AND id=?", (kind, key)).fetchone()
        if row is None:
            raise KeyError(f"{kind.capitalize()} not found.")
        return json.loads(row[0])

    def list(self, kind, include_deleted=False):
        with self.connect() as db:
            rows = db.execute("SELECT payload FROM records WHERE kind=? ORDER BY rowid DESC", (kind,)).fetchall()
        records = [json.loads(row[0]) for row in rows]
        if kind == "dataset" and not include_deleted:
            records = [record for record in records if not record.get("deleted_at")]
        return records

    def set_dataset_deleted(self, key, deleted):
        """Retain observations and references so deletion is reversible."""
        record = self.get("dataset", key)
        if deleted:
            record.setdefault("deleted_at", now())
        else:
            record.pop("deleted_at", None)
        return self.put("dataset", record)

    def add_dataset(self, data, columns, name, provenance):
        data = np.asarray(data, dtype=np.float64)
        if data.ndim != 2 or not 2 <= data.shape[0] <= 1_000_000 or not 1 <= data.shape[1] <= 100:
            raise ValueError("Use 2–1,000,000 rows and 1–100 variables.")
        if not np.isfinite(data).all():
            raise ValueError("All observations must be finite numeric values; missing values are not supported.")
        if np.any(np.var(data, axis=0) <= 1e-14):
            raise ValueError("Every variable must have nonzero variance.")
        if len(columns) != data.shape[1]:
            raise ValueError("The number of column names must match the data.")
        if not isinstance(name, str) or not 1 <= len(name) <= 120:
            raise ValueError("A sample name must contain 1–120 characters.")
        if any(not isinstance(c, dict) or not isinstance(c.get("id"), str) or not isinstance(c.get("label"), str)
               or not 1 <= len(c["id"]) <= 80 or not 1 <= len(c["label"]) <= 80 for c in columns):
            raise ValueError("Each dataset column needs a valid ID and name.")
        if len({c["id"] for c in columns}) != len(columns) or len({c["label"] for c in columns}) != len(columns):
            raise ValueError("Dataset column IDs and names must be unique.")
        if not isinstance(provenance, dict) or not isinstance(provenance.get("type"), str):
            raise ValueError("Dataset provenance must identify its type.")
        if provenance.get("graph"):
            from .models import Graph
            from .scoring import graph_arrays
            graph_arrays(Graph.model_validate(provenance["graph"]), columns)
        key = uuid.uuid4().hex
        path = self.root / "datasets" / f"{key}.npy"
        np.save(path, data, allow_pickle=False)
        return self.put("dataset", {"id": key, "name": name, "columns": columns,
            "n": data.shape[0], "p": data.shape[1], "created_at": now(),
            "sha256": hashlib.sha256(data.tobytes()).hexdigest(), "provenance": provenance,
            "preview": data[:8].tolist()})

    def dataset_path(self, key):
        self.get("dataset", key)
        return self.root / "datasets" / f"{key}.npy"

    def load_data(self, key):
        return np.load(self.dataset_path(key), allow_pickle=False)

    def import_csv(self, raw, name, header=True):
        if len(raw) > 50 * 1024 * 1024:
            raise ValueError("CSV uploads are limited to 50 MB.")
        try:
            text = raw.decode("utf-8-sig")
        except UnicodeDecodeError as exc:
            raise ValueError("Save the CSV with UTF-8 encoding.") from exc
        rows = csv.reader(io.StringIO(text))
        try:
            first = next(rows)
        except StopIteration:
            raise ValueError("The CSV is empty.")
        if not first or len(first) > 100:
            raise ValueError("Use 1–100 variables.")
        labels = [label.strip() for label in first] if header else [f"X{i+1}" for i in range(len(first))]
        if any(not label or len(label) > 80 for label in labels) or len(set(labels)) != len(labels):
            raise ValueError("Column names must be nonempty, unique, and at most 80 characters.")
        data = []
        if not header:
            rows = itertools.chain([first], rows)
        for line, row in enumerate(rows, start=2 if header else 1):
            if len(row) != len(labels):
                raise ValueError(f"CSV row {line} has {len(row)} fields; expected {len(labels)}.")
            try:
                data.append([float(value) for value in row])
            except ValueError as exc:
                raise ValueError(f"CSV row {line} contains a missing or nonnumeric value.") from exc
            if len(data) > 1_000_000:
                raise ValueError("Use at most 1,000,000 observations.")
        columns = [{"id": f"x{i+1}", "label": label} for i, label in enumerate(labels)]
        return self.add_dataset(data, columns, name, {"type": "imported", "header": header})
