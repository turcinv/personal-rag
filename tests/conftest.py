# test_queries.py is a manual integration smoke test — it needs a populated
# ChromaDB index and loads the embedding model, so it is not part of the offline
# unit suite. Keep pytest from collecting (and importing) it.
collect_ignore = ["test_queries.py"]

import pytest


def _real_model_available():
    """True iff the real embedding model can load without a network download.

    Set the HF hub offline flags, then attempt a load from the local cache. A
    cached model loads; an uncached one raises, and the real-model tests are
    skipped — preserving make test-unit's offline guarantee on a cold runner.
    """
    import os

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    try:
        from sentence_transformers import SentenceTransformer  # noqa: F401
    except Exception:
        return False
    try:
        SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        return True
    except Exception:
        return False


@pytest.fixture(scope="session")
def real_model():
    """Session-scoped real embedding model, or skip if unavailable offline."""
    if not _real_model_available():
        pytest.skip("real embedding model not cached locally (offline)")
    from rag.query import get_model

    return get_model("sentence-transformers/all-MiniLM-L6-v2")
