"""Tests for the memory-safe ``rag-stats`` CLI."""

import io
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from rag.stats import _infer_source_type


REPO_ROOT = Path(__file__).resolve().parents[1]


def _make_mock_store(records):
    store = MagicMock()
    store.count.return_value = len(records)
    store.name = "test_collection"

    store.page_sizes_seen = []

    def iter_records(page_size=10_000):
        store.page_sizes_seen.append(page_size)
        yield from records

    store.iter_records = iter_records
    return store


def _run_main(monkeypatch, records, *args):
    from rag.stats import main

    store = _make_mock_store(records)
    config = {
        "store": "chroma",
        "index_path": "/tmp/test",
        "collection_name": "test_collection",
    }
    with patch("rag.stats.load_config", return_value=config), \
         patch("rag.stats.setup_logging"), \
         patch("rag.stats.get_store", return_value=store):
        monkeypatch.setattr(sys, "argv", ["rag-stats", *args])
        captured = io.StringIO()
        monkeypatch.setattr(sys, "stdout", captured)
        main()
    return captured.getvalue(), store


def test_rag_stats_registered_as_console_script():
    """Keep the entry-point assertion compatible with Python 3.10."""
    pyproject = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert 'rag-stats = "rag.stats:main"' in pyproject


@pytest.mark.parametrize(
    ("metadata", "expected"),
    [
        ({"source": "", "path": "Knowledge/DevOps/Docker.md"}, "markdown"),
        ({"source": "pdf", "path": "book.pdf"}, "book/resource"),
        ({"source": "", "path": "book.pdf"}, "book/resource"),
        ({"source": "custom", "path": "orphan.txt"}, "other"),
        ({}, "other"),
    ],
)
def test_infer_source_type(metadata, expected):
    assert _infer_source_type(metadata) == expected


def test_stats_main_prints_total_and_source_breakdown(monkeypatch):
    records = [
        ("id1", "doc1", {"source": "", "path": "Knowledge/DevOps/Docker.md"}),
        ("id2", "doc2", {"source": "", "path": "Knowledge/DevOps/Docker.md"}),
        ("id3", "doc3", {"source": "", "path": "Knowledge/Python/FastAPI.md"}),
        ("id4", "doc4", {"source": "pdf", "path": "pythonfordevops.pdf"}),
        ("id5", "doc5", {"source": "pdf", "path": "tcpipillustrated.pdf"}),
    ]

    output, _store = _run_main(monkeypatch, records)

    assert "Collection: test_collection" in output
    assert "Total chunks: 5" in output
    assert "markdown" in output
    assert "3 chunks  (2 files)" in output
    assert "book/resource" in output
    assert "2 chunks  (2 files)" in output


def test_stats_main_handles_empty_index(monkeypatch):
    output, _store = _run_main(monkeypatch, [])
    assert "Total chunks: 0" in output
    assert "By source type:" not in output


def test_stats_main_uses_requested_page_size(monkeypatch):
    records = [("id1", "doc1", {"path": "Knowledge/DevOps/Docker.md"})]
    output, store = _run_main(monkeypatch, records, "--page-size", "5000")

    assert "Total chunks: 1" in output
    assert store.page_sizes_seen == [5000]


def test_stats_main_rejects_non_positive_page_size(monkeypatch):
    from rag.stats import main

    monkeypatch.setattr(sys, "argv", ["rag-stats", "--page-size", "0"])
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 2


def test_stats_main_collections_flag(monkeypatch):
    from rag.stats import main

    config = {"store": "chroma", "index_path": "/tmp/test"}
    with patch("rag.stats.load_config", return_value=config), \
         patch("rag.stats.setup_logging"), \
         patch(
             "rag.stats.list_collection_names",
             return_value={"obsidian_markdown", "wiki_lm", "test_coll"},
         ):
        monkeypatch.setattr(sys, "argv", ["rag-stats", "--collections"])
        captured = io.StringIO()
        monkeypatch.setattr(sys, "stdout", captured)
        main()

    output = captured.getvalue()
    assert "Collections (3):" in output
    assert output.index("obsidian_markdown") < output.index("test_coll") < output.index("wiki_lm")


def test_stats_module_uses_store_abstraction():
    text = (REPO_ROOT / "src" / "rag" / "stats.py").read_text(encoding="utf-8")
    assert "import chromadb" not in text
    assert "chromadb." not in text
