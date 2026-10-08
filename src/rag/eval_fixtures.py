"""Synthetic stale-trap fixtures for the eval harness.

Builds a throwaway, fully isolated store from synthetic "stale vs. current"
fixture pages and scores stale-trap rows against it through the real
chunking / embedding / :func:`rag.query.search` path — never touching the
production vault or Chroma collection. See ``.specs/stale-trap-fixtures.md``.

The fixture pages are synthetic (clearly marked ``synthetic: true`` /
``type: fixture`` and ``<Synthetic> ...`` titles) and are generated to real
``.md`` files inside a per-run temp directory, so ``extract_md_file`` runs on
real bytes on disk exactly as production does. The temp directory (vault +
Chroma store) is deleted when :func:`run_stale_trap` returns.

T2 is a deliberately-failing baseline (``baseline: True``): personal-rag has no
freshness/recency signal to prefer the newer of two equally-``processed`` notes,
so it is excluded from the pass count and reported separately. See §4.5.
"""

import tempfile
from pathlib import Path

from .query import search, get_model
from .store import ChromaStore
from .indexing import index_file_chunks
from .reconciliation import ReconciliationCatalog
from .extractors.markdown import extract_md_file
from .eval import first_hit_rank, stale_trap_passed

SYNTHETIC_MARKER = "SYNTHETIC FIXTURE"

#: The 6 synthetic page definitions (3 topics x stale/current). Each page is
#: short enough to stay a single chunk, so stale-trap rank assertions are
#: unambiguous. Content is paraphrased from real supersessions (verified by
#: Vít 2026-10-08); never copied verbatim from the real vault notes.
FIXTURE_PAGES = [
    # ── T1: parser-generator async job endpoint (needs-review -> processed) ──
    {
        "topic": "t1",
        "rel_path": "Synthetic/Stale-Trap/parser-generator-job-status-stale.md",
        "title": "<Synthetic> Parser Generator Job Status Endpoint (stale)",
        "status": "needs-review",
        "body": (
            f"**{SYNTHETIC_MARKER} \u2014 stale-trap test page, not a real vault note.**\n\n"
            "Historically, the parser-generator submitted a parse job "
            "asynchronously and polled `GET /api/check/{id}` for completion "
            "status, using an in-memory job store keyed by job id."
        ),
    },
    {
        "topic": "t1",
        "rel_path": "Synthetic/Stale-Trap/parser-generator-inline-current.md",
        "title": "<Synthetic> Parser Generator Inline Response (current)",
        "status": "processed",
        "body": (
            f"**{SYNTHETIC_MARKER} \u2014 stale-trap test page, not a real vault note.**\n\n"
            "As of parser-generator#3 (MR !12, merged 2026-07-28), "
            "`GET /api/check/{id}` and the in-memory job store were removed. "
            "`POST /api/submit` and `POST /api/generate/{id}` run the pipeline "
            "inline and return 200: no polling; `{id}` stays as a correlation "
            "token (a UUID, validated and echoed back in error bodies, but no "
            "longer used to poll for status)."
        ),
    },
    # ── T2: alert-generator production infra (processed -> processed) baseline ──
    {
        "topic": "t2",
        "rel_path": "Synthetic/Stale-Trap/alert-generator-runpod-stale.md",
        "title": "<Synthetic> Alert Generator RunPod Production (stale)",
        "status": "processed",
        "body": (
            f"**{SYNTHETIC_MARKER} \u2014 stale-trap test page, not a real vault note.**\n\n"
            "The alert generator runs in production on RunPod serverless GPU "
            "endpoints: inference jobs are submitted to a RunPod serverless "
            "worker and the production backend talks to the RunPod HTTP API."
        ),
    },
    {
        "topic": "t2",
        "rel_path": "Synthetic/Stale-Trap/alert-generator-eks-current.md",
        "title": "<Synthetic> Alert Generator EKS Deployment (current)",
        "status": "processed",
        "body": (
            f"**{SYNTHETIC_MARKER} \u2014 stale-trap test page, not a real vault note.**\n\n"
            "The alert and parser generator now run on AWS EKS (`lm-eks`) with a "
            "g5.xlarge GPU node pool. Production inference serves from the EKS "
            "cluster's GPU nodes, replacing the earlier serverless deployment."
        ),
    },
    # ── T3: grammar enforcement library (needs-review -> processed) ──
    {
        "topic": "t3",
        "rel_path": "Synthetic/Stale-Trap/grammar-transformers-cfg-stale.md",
        "title": "<Synthetic> Grammar Enforcement transformers_cfg (stale)",
        "status": "needs-review",
        "body": (
            f"**{SYNTHETIC_MARKER} \u2014 stale-trap test page, not a real vault note.**\n\n"
            "The alert generator enforces its EBNF grammar during decoding with "
            "`transformers_cfg`, plugging a custom `LogitsProcessor` into the "
            "generation loop to constrain token logits to the grammar."
        ),
    },
    {
        "topic": "t3",
        "rel_path": "Synthetic/Stale-Trap/grammar-vllm-xgrammar-current.md",
        "title": "<Synthetic> Grammar Enforcement vLLM xgrammar (current)",
        "status": "processed",
        "body": (
            f"**{SYNTHETIC_MARKER} \u2014 stale-trap test page, not a real vault note.**\n\n"
            "Grammar-constrained decoding now uses vLLM with xgrammar for "
            "structured output enforcement, replacing the earlier "
            "`transformers_cfg` `LogitsProcessor` approach."
        ),
    },
]

#: Stale-trap golden rows, one per topic. Queries are phrased in the STALE
#: page's own vocabulary so the trap is adversarial (the stale page would win
#: absent the needs-review filter — proven by the teeth tests' precondition).
#: ``baseline: True`` marks the T2 row, which is a documented expected-fail
#: (excluded from the pass count); see §4.5.
STALE_TRAP_GOLDEN = [
    {
        "query": "How does the frontend poll for parser job status?",
        "expected": ["<Synthetic> Parser Generator Inline Response (current)"],
        "must_not": ["<Synthetic> Parser Generator Job Status Endpoint (stale)"],
        "kind": "stale-trap",
    },
    {
        "query": "Where does the alert generator run on RunPod serverless in production?",
        "expected": ["<Synthetic> Alert Generator EKS Deployment (current)"],
        "must_not": ["<Synthetic> Alert Generator RunPod Production (stale)"],
        "kind": "stale-trap",
        "baseline": True,
    },
    {
        "query": "How is the EBNF grammar enforced with transformers_cfg and a LogitsProcessor?",
        "expected": ["<Synthetic> Grammar Enforcement vLLM xgrammar (current)"],
        "must_not": ["<Synthetic> Grammar Enforcement transformers_cfg (stale)"],
        "kind": "stale-trap",
    },
]


def _write_fixture_vault(vault_dir: Path):
    """Write FIXTURE_PAGES out as real .md files (frontmatter + body)."""
    for page in FIXTURE_PAGES:
        path = vault_dir / page["rel_path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        frontmatter = (
            "---\n"
            f'title: "{page["title"]}"\n'
            f"status: {page['status']}\n"
            "synthetic: true\n"
            "type: fixture\n"
            "---\n\n"
        )
        path.write_text(frontmatter + page["body"] + "\n", encoding="utf-8")


def build_fixture_store(config, tmp_dir, model):
    """Build and populate an isolated fixture store under ``tmp_dir``.

    Writes the synthetic pages to ``tmp_dir/vault`` and ingests them into a
    fresh Chroma collection under ``tmp_dir/chroma`` via the real
    ``extract_md_file`` / ``index_file_chunks`` path. The store is never opened
    through ``open_store()``/the production config, so no provenance
    fingerprint check applies. ``model`` is the already-loaded embedding model
    (not reloaded). Returns the populated store.
    """
    tmp_dir = Path(tmp_dir)
    vault_dir = tmp_dir / "vault"
    index_path = tmp_dir / "chroma"
    vault_dir.mkdir(parents=True, exist_ok=True)
    index_path.mkdir(parents=True, exist_ok=True)

    _write_fixture_vault(vault_dir)

    store = ChromaStore(str(index_path), "stale_trap_fixtures")
    store.ensure("stale_trap_fixtures")

    max_chars = int(config.get("chunk_max_chars", 1200))
    overlap = int(config.get("chunk_overlap_chars", 150))
    embed_batch = int(config.get("embedding_batch_size", 16))

    # Empty store -> empty catalog -> every fixture chunk classifies as `new`.
    with ReconciliationCatalog.from_store(store, index_path) as reconciliation:
        for page in FIXTURE_PAGES:
            md_file = vault_dir / page["rel_path"]
            ids, docs, metas, error = extract_md_file(
                md_file, vault_dir, config, max_chars, overlap
            )
            if error:
                raise RuntimeError(f"fixture extract failed: {error}")
            index_file_chunks(
                ids, docs, metas, reconciliation, model, "cpu", embed_batch, store
            )
    return store


#: Note recorded on a result row when hybrid scoring was requested but skipped.
HYBRID_SKIPPED_NOTE = "skipped (fixture store has no lexical index)"


def _score_row(store, row, *, config, model, n, rerank, hybrid, include_unreviewed=False):
    """Run one stale-trap row against the fixture store and build its result dict.

    The fixture store has no lexical index built (``make build-lexical`` is
    never run against it) and its own dense index carries no vector-generation
    provenance, so the client-side BM25 hybrid path (``query.py``) would either
    raise (``read_index_state`` returns ``None``) or read the *production*
    lexical index via ``get_lexical(config, ...)``. Neither is acceptable here,
    so when ``hybrid`` is requested the row is scored dense-only and the skip is
    recorded in ``result["hybrid"]`` instead of silently ignored.
    """
    records = search(
        row["query"], n, store=store, config=config, model=model,
        rerank=rerank, hybrid=False, include_unreviewed=include_unreviewed,
    )
    expected, must_not = row["expected"], row.get("must_not", [])
    return {
        "query": row["query"],
        "expected": expected,
        "must_not": must_not,
        "expected_rank": first_hit_rank(records, expected),
        "must_not_rank": first_hit_rank(records, must_not) if must_not else None,
        "passed": stale_trap_passed(records, expected, must_not),
        "baseline": bool(row.get("baseline", False)),
        "hybrid": HYBRID_SKIPPED_NOTE if hybrid else None,
    }


def run_stale_trap(config, model=None, *, n=10, rerank=False, hybrid=False):
    """Build an isolated fixture store and score every stale-trap row against it.

    Returns ``{"rows": [...], "baseline_rows": [...]}`` where ``rows`` are the
    pass-counted trap rows and ``baseline_rows`` are documented expected-fails
    (T2), excluded from the pass count. The temp directory (vault + store) is
    created and destroyed within this call. ``model`` is loaded from ``config``
    if not supplied.
    """
    if model is None:
        model_name = config.get("embedding_model", "sentence-transformers/all-MiniLM-L6-v2")
        model = get_model(model_name, revision=config.get("embedding_revision") or None)

    rows, baseline_rows = [], []
    with tempfile.TemporaryDirectory(prefix="rag-stale-trap-") as tmp_dir:
        store = build_fixture_store(config, tmp_dir, model)
        for row in STALE_TRAP_GOLDEN:
            scored = _score_row(
                store, row, config=config, model=model, n=n,
                rerank=rerank, hybrid=hybrid,
            )
            if scored["baseline"]:
                baseline_rows.append(scored)
            else:
                rows.append(scored)
    return {"rows": rows, "baseline_rows": baseline_rows}
