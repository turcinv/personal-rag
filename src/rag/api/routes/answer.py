"""Answer surface: POST /answer — retrieval-augmented generation.

Sits above the retrieval core: runs the same ``build_where`` + ``search`` path
as ``/query`` to fetch the relevant chunks, then asks the configured
``Generator`` (loaded once in the app lifespan) for a grounded, cited answer.
JWT-protected like the rest of the authenticated surface.

Two deliberate guards:
  * If generation is not configured (no ``generation`` block / no API key), the
    generator is ``None`` in app state and this endpoint returns 503 — ``/query``
    keeps working regardless.
  * If retrieval returns no chunks, the endpoint returns a fixed "no relevant
    context" answer WITHOUT calling the LLM, so it can never hallucinate from an
    empty context.
"""

import logging

from fastapi import APIRouter, Depends, HTTPException

from ... import query as rag_query
from ...generation import GenerationError
from ..auth import require_jwt
from ..deps import get_rag_state
from ..schemas import AnswerRequest, AnswerResponse, Citation

logger = logging.getLogger("rag")

router = APIRouter()

NO_CONTEXT_ANSWER = (
    "I couldn't find anything relevant in the knowledge base to answer that, "
    "so I won't guess. Try rephrasing, or widen any filters you applied."
)


def _citations(records: list) -> list:
    """Map ``search()`` records to Citation models, 1-based to match ``[n]``."""
    out = []
    for i, r in enumerate(records, start=1):
        meta = r.get("metadata") or {}
        out.append(
            Citation(
                n=i,
                title=meta.get("title"),
                path=meta.get("path"),
                domain=meta.get("domain"),
                distance=r.get("distance"),
                rerank_score=r.get("rerank_score"),
            )
        )
    return out


@router.post("/answer", response_model=AnswerResponse, tags=["answer"])
def answer(
    request: AnswerRequest,
    state: dict = Depends(get_rag_state),
    _claims: dict = Depends(require_jwt),
) -> AnswerResponse:
    """Retrieve, then synthesize a grounded answer with inline [n] citations."""
    generator = state.get("generator")
    if generator is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "Answer generation is not configured on this server. Set a "
                "`generation` block in the active config and export the provider "
                "API key, then restart. Retrieval (/query) is unaffected."
            ),
        )

    f = request.filters
    where = rag_query.build_where(
        domain=f.domain if f else None,
        type_=f.type if f else None,
        source=f.source if f else None,
        confidence=f.confidence if f else None,
        subdomain=f.subdomain if f else None,
        status=f.status if f else None,
    )

    # rerank omitted (None) → fall back to the profile's rerank_default.
    rerank = (
        request.rerank
        if request.rerank is not None
        else rag_query.rerank_default(state["config"])
    )

    records = rag_query.search(
        request.query,
        n_results=request.n_results,
        filters=where,
        tags=f.tags if f else None,
        config=state["config"],
        model=state["model"],
        store=state["store"],
        rerank=rerank,
    )

    reranked = rerank and any("rerank_score" in r for r in records)

    # No context → fixed refusal, never call the LLM (no hallucination surface).
    if not records:
        return AnswerResponse(
            query=request.query,
            answer=NO_CONTEXT_ANSWER,
            grounded=False,
            provider=generator.provider,
            model=generator.model,
            reranked=reranked,
            citations=[],
            usage=None,
        )

    try:
        result = generator.generate(
            request.query,
            records,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
        )
    except GenerationError as exc:
        # Upstream LLM failure is a bad-gateway condition, not our 500.
        logger.warning("generation failed: %s", exc)
        raise HTTPException(status_code=502, detail=f"Answer generation failed: {exc}")

    return AnswerResponse(
        query=request.query,
        answer=result.answer,
        grounded=True,
        provider=generator.provider,
        model=result.model,
        reranked=reranked,
        citations=_citations(records),
        usage=result.usage,
    )
