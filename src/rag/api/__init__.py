"""HTTP backend for personal-rag.

A FastAPI app that loads the embedding model, ChromaDB collection, and
cross-encoder reranker exactly once at process startup (see ``app.lifespan``)
and shares them across requests. This is the whole reason the API exists instead
of shelling out to the ``rag-query`` CLI per call. Entry point: ``rag-serve``.
"""
