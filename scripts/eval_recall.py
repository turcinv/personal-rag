#!/usr/bin/env python
"""Thin CLI shim for the recall@k / MRR eval harness (see rag.eval).

Run via `make eval` / `make jetson-eval`, or directly:
    .venv/bin/python scripts/eval_recall.py --label baseline --out tests/eval/baseline.json
"""

from rag.eval import main

if __name__ == "__main__":
    main()
