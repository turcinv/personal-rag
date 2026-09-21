"""Disk-backed index reconciliation with bounded Python memory."""

from __future__ import annotations

import json
import os
import sqlite3
import tempfile
from pathlib import Path
from typing import Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple


class ReconciliationCatalog:
    """Temporary SQLite catalog for one index run.

    Existing metadata is paged from the store before mutations begin. Later
    classification, preservation, planning, and deletion use indexed SQL rather
    than retaining a corpus-sized Python dictionary or ID set.
    """

    def __init__(self, path: Path, connection: sqlite3.Connection, existing_count: int):
        self.path = path
        self.connection = connection
        self.existing_count = existing_count

    @classmethod
    def from_store(
        cls,
        store,
        index_path: Path,
        page_size: int = 5_000,
        insert_batch_size: int = 1_000,
    ) -> "ReconciliationCatalog":
        parent = Path(index_path).expanduser().resolve().parent
        parent.mkdir(parents=True, exist_ok=True)
        descriptor, raw_path = tempfile.mkstemp(
            prefix=".rag-reconcile-",
            suffix=".sqlite3",
            dir=str(parent),
        )
        os.close(descriptor)
        path = Path(raw_path)
        connection = sqlite3.connect(str(path))
        try:
            connection.execute("PRAGMA temp_store=FILE")
            connection.execute("PRAGMA cache_size=-4096")
            connection.execute("PRAGMA journal_mode=OFF")
            connection.execute("PRAGMA synchronous=OFF")
            connection.executescript(
                """
                CREATE TABLE chunks (
                    chunk_id TEXT PRIMARY KEY,
                    metadata_json TEXT NOT NULL,
                    original_source_id TEXT,
                    current_source_id TEXT,
                    file_id TEXT,
                    path TEXT,
                    was_existing INTEGER NOT NULL,
                    seen INTEGER NOT NULL DEFAULT 0
                );
                CREATE INDEX chunks_original_source
                    ON chunks(original_source_id, was_existing, seen);
                CREATE INDEX chunks_original_file
                    ON chunks(original_source_id, file_id, was_existing);
                CREATE INDEX chunks_current_source
                    ON chunks(current_source_id, seen);
                """
            )
            expected = int(store.count())
            loaded = 0
            batch = []
            for chunk_id, metadata in store.iter_metadata(page_size=page_size):
                metadata = dict(metadata or {})
                source_id = metadata.get("source_id")
                batch.append(
                    (
                        chunk_id,
                        _metadata_json(metadata),
                        source_id,
                        source_id,
                        metadata.get("file_id"),
                        metadata.get("path"),
                        1,
                        0,
                    )
                )
                if len(batch) >= insert_batch_size:
                    connection.executemany(_INSERT_SQL, batch)
                    loaded += len(batch)
                    batch.clear()
            if batch:
                connection.executemany(_INSERT_SQL, batch)
                loaded += len(batch)
            connection.commit()
            if loaded != expected:
                raise RuntimeError(
                    f"Index changed while reconciliation was paging metadata: "
                    f"expected {expected} chunks, loaded {loaded}"
                )
            return cls(path, connection, loaded)
        except Exception:
            connection.close()
            path.unlink(missing_ok=True)
            raise

    def __enter__(self) -> "ReconciliationCatalog":
        return self

    def __exit__(self, _exc_type, _exc, _traceback) -> None:
        self.close()

    def close(self) -> None:
        if self.connection is not None:
            self.connection.close()
            self.connection = None  # type: ignore[assignment]  # sentinel after close(); guarded above
        self.path.unlink(missing_ok=True)

    def classify_and_mark(
        self,
        ids: Sequence[str],
        metadatas: Sequence[Mapping[str, object]],
    ) -> Tuple[List[int], List[int]]:
        """Mark one file's chunks seen and return new/metadata-update indices."""
        new_indices: List[int] = []
        updated_indices: List[int] = []
        cursor = self.connection.cursor()
        for index, (chunk_id, metadata_value) in enumerate(zip(ids, metadatas)):
            metadata = dict(metadata_value)
            encoded = _metadata_json(metadata)
            row = cursor.execute(
                "SELECT metadata_json FROM chunks WHERE chunk_id = ?",
                (chunk_id,),
            ).fetchone()
            if row is None:
                new_indices.append(index)
                cursor.execute(
                    _INSERT_SQL,
                    (
                        chunk_id,
                        encoded,
                        None,
                        metadata.get("source_id"),
                        metadata.get("file_id"),
                        metadata.get("path"),
                        0,
                        1,
                    ),
                )
                continue
            if row[0] != encoded:
                updated_indices.append(index)
            cursor.execute(
                """
                UPDATE chunks
                SET metadata_json = ?, current_source_id = ?, file_id = ?,
                    path = ?, seen = 1
                WHERE chunk_id = ?
                """,
                (
                    encoded,
                    metadata.get("source_id"),
                    metadata.get("file_id"),
                    metadata.get("path"),
                    chunk_id,
                ),
            )
        self.connection.commit()
        return new_indices, updated_indices

    def preserve_file(self, source_id: str, file_id: str) -> int:
        cursor = self.connection.execute(
            """
            UPDATE chunks SET seen = 1
            WHERE was_existing = 1 AND original_source_id = ? AND file_id = ?
            """,
            (source_id, file_id),
        )
        self.connection.commit()
        return cursor.rowcount

    def source_counts(self, source_id: str) -> Tuple[int, int, int]:
        existing = self._scalar(
            "SELECT COUNT(*) FROM chunks WHERE was_existing = 1 AND original_source_id = ?",
            (source_id,),
        )
        seen = self._scalar(
            "SELECT COUNT(*) FROM chunks WHERE seen = 1 AND current_source_id = ?",
            (source_id,),
        )
        stale = self._scalar(
            """
            SELECT COUNT(*) FROM chunks
            WHERE was_existing = 1 AND original_source_id = ? AND seen = 0
            """,
            (source_id,),
        )
        return existing, seen, stale

    def owned_sources(self) -> Iterator[Tuple[str, int]]:
        cursor = self.connection.execute(
            """
            SELECT original_source_id, COUNT(*) FROM chunks
            WHERE was_existing = 1 AND original_source_id IS NOT NULL
            GROUP BY original_source_id ORDER BY original_source_id
            """
        )
        yield from cursor

    def legacy_count(self) -> int:
        return self._scalar(
            """
            SELECT COUNT(*) FROM chunks
            WHERE was_existing = 1 AND original_source_id IS NULL
            """
        )

    def iter_stale_batches(
        self,
        source_id: str,
        batch_size: int = 500,
    ) -> Iterator[List[str]]:
        cursor = self.connection.execute(
            """
            SELECT chunk_id FROM chunks
            WHERE was_existing = 1 AND original_source_id = ? AND seen = 0
            ORDER BY chunk_id
            """,
            (source_id,),
        )
        while True:
            rows = cursor.fetchmany(batch_size)
            if not rows:
                break
            yield [row[0] for row in rows]

    def _scalar(self, sql: str, parameters: Iterable[object] = ()) -> int:
        row = self.connection.execute(sql, tuple(parameters)).fetchone()
        return int(row[0])


_INSERT_SQL = """
INSERT INTO chunks (
    chunk_id, metadata_json, original_source_id, current_source_id,
    file_id, path, was_existing, seen
) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
"""


def _metadata_json(metadata: Mapping[str, object]) -> str:
    return json.dumps(metadata, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
