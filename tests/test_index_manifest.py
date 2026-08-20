"""Tests for disk-backed source-scoped prune planning and manifests."""

import json
import tempfile
from pathlib import Path

import pytest

from rag.index_manifest import (
    apply_prune_plan,
    build_prune_plan,
    manifest_directory,
    write_run_manifest,
)
from rag.indexing import SourceRunResult
from rag.reconciliation import ReconciliationCatalog


def _result(source_id, state="completed", new=0, updated=0, failed=0, empty=0, files=1):
    return SourceRunResult(
        source_id=source_id,
        state=state,
        file_count=files,
        succeeded_files=files - failed - empty,
        failed_files=failed,
        empty_files=empty,
        total_chunks=0,
        new_chunks=new,
        updated_chunks=updated,
        detail="degraded" if state == "degraded" else "",
    )


def _existing():
    return {
        "a1": {"source_id": "markdown:vault", "file_id": "a.md"},
        "a2": {"source_id": "markdown:vault", "file_id": "deleted.md"},
        "b1": {"source_id": "json:catalog", "file_id": "book.json"},
        "legacy": {"path": "old.md"},
    }


class MetadataStore:
    def __init__(self, records):
        self.records = records

    def count(self):
        return len(self.records)

    def iter_metadata(self, page_size=5_000):
        del page_size
        yield from self.records.items()


def _catalog(records):
    root = Path(tempfile.mkdtemp())
    return ReconciliationCatalog.from_store(MetadataStore(records), root / "index")


def _mark(reconciliation, records, ids):
    reconciliation.classify_and_mark(ids, [records[chunk_id] for chunk_id in ids])


def test_completed_source_prunes_only_its_owned_stale_chunks():
    records = _existing()
    with _catalog(records) as reconciliation:
        _mark(reconciliation, records, ["a1"])
        plan = build_prune_plan(
            reconciliation,
            [
                _result("markdown:vault"),
                _result("json:catalog", state="missing", files=0),
            ],
            max_fraction=1.0,
            max_chunks=100,
        )

        assert plan.approved_deletion_count == 1
        assert plan.legacy_chunks_preserved == 1
        json_plan = next(
            source for source in plan.sources if source.source_id == "json:catalog"
        )
        assert json_plan.blocked
        assert json_plan.preserved_chunks == 1


def test_degraded_source_never_deletes_failed_or_empty_file_chunks():
    with _catalog(_existing()) as reconciliation:
        plan = build_prune_plan(
            reconciliation,
            [_result("json:catalog", state="degraded", failed=1, files=1)],
            max_fraction=1.0,
            max_chunks=100,
        )
        source = plan.sources[0]
        assert source.blocked
        assert source.proposed_deletion_count == 1
        assert plan.approved_deletion_count == 0


def test_empty_source_requires_explicit_override():
    result = _result("markdown:vault", state="empty", files=0)
    with _catalog(_existing()) as reconciliation:
        blocked = build_prune_plan(
            reconciliation, [result], max_fraction=1.0, max_chunks=100
        )
        allowed = build_prune_plan(
            reconciliation,
            [result],
            max_fraction=1.0,
            max_chunks=100,
            allow_empty_source_prune=True,
        )

        assert blocked.approved_deletion_count == 0
        assert allowed.approved_deletion_count == 2


def test_large_deletion_is_blocked_until_explicitly_allowed():
    records = _existing()
    with _catalog(records) as reconciliation:
        _mark(reconciliation, records, ["a1"])
        result = _result("markdown:vault")
        blocked = build_prune_plan(
            reconciliation, [result], max_fraction=0.25, max_chunks=100
        )
        allowed = build_prune_plan(
            reconciliation,
            [result],
            max_fraction=0.25,
            max_chunks=100,
            allow_large_prune=True,
        )

        assert blocked.approved_deletion_count == 0
        assert "threshold exceeded" in blocked.sources[0].reason
        assert allowed.approved_deletion_count == 1


def test_removed_configuration_preserves_owned_source():
    records = _existing()
    with _catalog(records) as reconciliation:
        _mark(reconciliation, records, ["a1", "a2"])
        plan = build_prune_plan(
            reconciliation,
            [_result("markdown:vault")],
            max_fraction=1.0,
            max_chunks=100,
        )
        orphan = next(
            source for source in plan.sources if source.source_id == "json:catalog"
        )
        assert orphan.state == "not-configured"
        assert orphan.blocked
        assert plan.approved_deletion_count == 0


def test_apply_prune_plan_streams_only_approved_ids_in_batches():
    with _catalog(_existing()) as reconciliation:
        plan = build_prune_plan(
            reconciliation,
            [_result("markdown:vault")],
            max_fraction=1.0,
            max_chunks=100,
            allow_large_prune=True,
        )

        class Store:
            def __init__(self):
                self.calls = []

            def delete(self, ids):
                self.calls.append(ids)

        store = Store()
        count = apply_prune_plan(store, plan, reconciliation, batch_size=1)
        assert count == 2
        assert store.calls == [["a1"], ["a2"]]


def test_write_run_manifest_is_parseable_and_keeps_source_decisions(tmp_path):
    records = _existing()
    with _catalog(records) as reconciliation:
        _mark(reconciliation, records, ["a1", "a2"])
        plan = build_prune_plan(
            reconciliation,
            [_result("markdown:vault")],
            max_fraction=1.0,
            max_chunks=100,
        )
    config = {"manifest_path": str(tmp_path / "manifests")}

    path = write_run_manifest(config, "collection", plan)
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert path.parent == tmp_path / "manifests" / "collection"
    assert payload["run_id"] == plan.run_id
    assert payload["legacy_chunks_preserved"] == 1
    assert payload["sources"][0]["source_id"] == "markdown:vault"
    assert "proposed_deletions" not in payload["sources"][0]


def test_default_manifest_directory_is_sibling_of_index(tmp_path):
    config = {"index_path": str(tmp_path / "chroma")}
    assert manifest_directory(config, "c") == tmp_path / "chroma_manifests" / "c"


def test_duplicate_source_results_are_rejected():
    with _catalog({}) as reconciliation:
        with pytest.raises(ValueError, match="Duplicate source result id.*json:catalog"):
            build_prune_plan(
                reconciliation,
                [_result("json:catalog"), _result("json:catalog")],
                max_fraction=1.0,
                max_chunks=100,
            )


def test_large_reconciliation_uses_lazy_metadata_and_bounded_delete_batches(tmp_path):
    class LazyStore:
        total = 12_345

        def count(self):
            return self.total

        def iter_metadata(self, page_size=5_000):
            assert page_size == 257
            for index in range(self.total):
                yield f"id-{index:05d}", {
                    "source_id": "json:catalog",
                    "file_id": f"file-{index // 10}.json",
                }

        def snapshot(self):
            raise AssertionError("full snapshots are forbidden")

    reconciliation = ReconciliationCatalog.from_store(
        LazyStore(), tmp_path / "index", page_size=257, insert_batch_size=113
    )
    with reconciliation:
        plan = build_prune_plan(
            reconciliation,
            [_result("json:catalog")],
            max_fraction=1.0,
            max_chunks=20_000,
        )

        class DeleteStore:
            def __init__(self):
                self.count = 0

            def delete(self, ids):
                assert len(ids) <= 500
                self.count += len(ids)

        store = DeleteStore()
        deleted = apply_prune_plan(store, plan, reconciliation, batch_size=500)

        assert deleted == LazyStore.total
        assert store.count == LazyStore.total
