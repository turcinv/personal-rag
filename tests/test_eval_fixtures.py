"""Tests for the synthetic stale-trap fixtures (``rag.eval_fixtures``).

Structural tests (§6.1) run with no model and cost nothing in CI. The harness
pass test (§6.2) and the teeth tests (§6.3) load the real embedding model and
are gated behind the ``real_model`` fixture (skipped when the model is not
cached offline). See ``.specs/stale-trap-fixtures.md``.
"""

import tempfile

import pytest

from rag.eval_fixtures import (
    FIXTURE_PAGES,
    STALE_TRAP_GOLDEN,
    SYNTHETIC_MARKER,
    build_fixture_store,
    run_stale_trap,
)
from rag.eval import first_hit_rank, stale_trap_passed
from rag.query import search

# Minimal config the fixture harness needs — deliberately NOT load_config(), so
# the test never depends on the real config.yaml (which points at the
# production vault/index). default_excluded_status mirrors config.yaml:81.
FIXTURE_CONFIG = {
    "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
    "chunk_max_chars": 1200,
    "chunk_overlap_chars": 150,
    "embedding_batch_size": 16,
    "default_excluded_status": ["needs-review"],
}


# ── §6.1 structural tests (no model, fast) ───────────────────────────────────

def test_fixture_pages_shape():
    assert len(FIXTURE_PAGES) == 6
    topics = {}
    for page in FIXTURE_PAGES:
        topics.setdefault(page["topic"], []).append(page)
        assert page["title"].startswith("<Synthetic> ")
        assert SYNTHETIC_MARKER in page["body"]
        assert page["status"] in {"needs-review", "processed"}
    assert set(topics) == {"t1", "t2", "t3"}
    for topic, pages in topics.items():
        assert len(pages) == 2, topic
        statuses = sorted(p["status"] for p in pages)
        # exactly one "current" (processed) page per topic
        assert statuses.count("processed") >= 1, topic


def test_fixture_pages_meet_status_requirements():
    """>=1 needs-review stale page and >=1 processed stale page across topics."""
    stale_statuses = []
    for row in STALE_TRAP_GOLDEN:
        must_not_title = row["must_not"][0]
        page = next(p for p in FIXTURE_PAGES if p["title"] == must_not_title)
        stale_statuses.append(page["status"])
    assert "needs-review" in stale_statuses  # T1/T3
    assert "processed" in stale_statuses     # T2 baseline


def test_golden_rows_shape_and_cross_reference():
    assert len(STALE_TRAP_GOLDEN) == 3
    titles = {p["title"] for p in FIXTURE_PAGES}
    baseline_count = 0
    for row in STALE_TRAP_GOLDEN:
        assert row["kind"] == "stale-trap"
        assert row["expected"] and isinstance(row["expected"], list)
        assert row["must_not"] and isinstance(row["must_not"], list)
        # every expected/must_not substring must resolve to a real fixture title
        for sub in row["expected"] + row["must_not"]:
            assert any(sub in t for t in titles), sub
        if row.get("baseline"):
            baseline_count += 1
    assert baseline_count == 1  # only T2 is a baseline row


# ── §6.2 end-to-end harness test (real model, real ChromaStore, temp dirs) ────

def test_run_stale_trap_t1_t3_pass_on_current_code(real_model):
    result = run_stale_trap(FIXTURE_CONFIG, real_model)
    assert len(result["rows"]) == 2          # T1 + T3, pass-counted
    assert len(result["baseline_rows"]) == 1  # T2, excluded
    assert all(r["passed"] for r in result["rows"]), result["rows"]


# ── §6.3 teeth tests — prove the scoring isn't vacuous ────────────────────────

def _row_for(topic_must_not_substr):
    return next(r for r in STALE_TRAP_GOLDEN
               if topic_must_not_substr in r["must_not"][0])


def _teeth(real_model, must_not_substr):
    """Precondition (stale ranks #1 unfiltered) + filter-on pass + filter-off fail."""
    row = _row_for(must_not_substr)
    with tempfile.TemporaryDirectory(prefix="rag-teeth-") as tmp:
        store = build_fixture_store(FIXTURE_CONFIG, tmp, real_model)
        q, expected, must_not = row["query"], row["expected"], row["must_not"]

        # Precondition: with the needs-review filter DISABLED, the stale page
        # ranks #1 — proving the query is genuinely adversarial. If this fails,
        # sharpen the fixture wording, never weaken this assertion (§6.3).
        unfiltered = search(q, 10, store=store, config=FIXTURE_CONFIG,
                            model=real_model, include_unreviewed=True)
        assert first_hit_rank(unfiltered, must_not) == 1, \
            "precondition: stale page must rank #1 with the filter off"

        # Filter ON (default): the trap passes.
        filtered = search(q, 10, store=store, config=FIXTURE_CONFIG,
                          model=real_model)
        assert stale_trap_passed(filtered, expected, must_not) is True

        # Filter OFF: the trap fails (meaningful only because of the precondition).
        assert stale_trap_passed(unfiltered, expected, must_not) is False


def test_t1_needs_review_trap_has_teeth(real_model):
    _teeth(real_model, "Parser Generator Job Status Endpoint (stale)")


def test_t3_needs_review_trap_has_teeth(real_model):
    _teeth(real_model, "Grammar Enforcement transformers_cfg (stale)")


@pytest.mark.xfail(strict=True, reason="personal-rag has no freshness/recency "
                   "signal between two processed notes — see "
                   ".specs/stale-trap-fixtures.md §4.5")
def test_t2_baseline_is_expected_to_fail(real_model):
    row = _row_for("Alert Generator RunPod Production (stale)")
    with tempfile.TemporaryDirectory(prefix="rag-t2-") as tmp:
        store = build_fixture_store(FIXTURE_CONFIG, tmp, real_model)
        records = search(row["query"], 10, store=store, config=FIXTURE_CONFIG,
                        model=real_model)
        assert stale_trap_passed(records, row["expected"], row["must_not"]) is True
