# RAG Project Fix Summary — Cryptography Dependency

> **HISTORICAL — written 2026-06-05 (the file originally carried no date). ARCHIVED 2026-07-28.**
> The conclusion still stands: `cryptography>=3.1` is required for AES-encrypted PDFs and
> is declared in `requirements-direct.txt`, `requirements-jetson.txt` and `pyproject.toml`.
> Everything procedural is stale — do **not** run the commands below:
> - The decrypt call moved from `src/rag/indexer.py` to `src/rag/extractors/pdf.py`.
> - **`rm -rf chroma_db` is harmful now.** Indexing is incremental and content-hashed;
>   deleting the store discards ~202k chunks and forces a full re-embed (hours on the
>   Jetson) to reach the same state. Just re-run `make index`.
> - The repo path is `/Users/turcinv/Documents/personal-rag`, and commands must use
>   `.venv/bin/...`, never bare `pip` / `python`.

## Summary

Some PDFs in the book and resource libraries (e.g. `begintocodewithpython.pdf`) are
AES-encrypted with an owner password but no user password. pypdf delegates AES decryption
to the `cryptography` package, which was not previously included as a dependency.
Without it, these PDFs were skipped during indexing with:

```
SKIP begintocodewithpython.pdf: cryptography>=3.1 is required for AES algorithm
```

The fix adds `cryptography>=3.1` to all dependency files and calls `reader.decrypt("")`
when pypdf reports a PDF as encrypted, transparently unlocking owner-password-only PDFs.

---

## Files Changed

| File | Change |
|---|---|
| `requirements-direct.txt` | Added `cryptography>=3.1` |
| `requirements.txt` | Regenerated lockfile — pins `cryptography==48.0.0`, `cffi==2.0.0`, `pycparser==3.0` |
| `requirements-jetson.txt` | Added `cryptography>=3.1` |
| `src/rag/indexer.py` | Added `reader.decrypt("")` after `PdfReader()` when `reader.is_encrypted` |
| `README.md` | Updated Known Issues — encrypted PDFs are now supported |
| `CLAUDE.md` | Added `cryptography` to Key dependencies table |

---

## Why cryptography is needed

pypdf handles RC4-encrypted PDFs in pure Python, but delegates AES-128 and AES-256
decryption to the `cryptography` library. Many commercially published PDFs are locked
with an owner password (restricts printing/copying) but no user password — they open
freely in any PDF reader but require `cryptography` to decrypt programmatically.

The decrypt call uses an empty string, which matches the blank user password on these PDFs:

```python
reader = PdfReader(str(pdf_path))
if reader.is_encrypted:
    reader.decrypt("")
```

PDFs that require a non-blank password will still be skipped with a clear warning;
the blank-password attempt is silent on failure and does not affect non-encrypted PDFs.

---

## Verification Commands

```bash
cd /Users/turcinv/Documents/GitHub/personal-rag

# Install / refresh deps (macOS)
uv pip install --python .venv/bin/python -r requirements.txt

# Confirm cryptography is importable
.venv/bin/python -c "import cryptography; print('cryptography OK', cryptography.__version__)"

# Re-index from scratch
rm -rf chroma_db
.venv/bin/rag-index 2>&1 | tee index_run.log

# Check for any remaining skipped files
grep -i "SKIP" index_run.log
```

For Jetson (run on the Jetson):

```bash
cd ~/personal-rag

pip install -r requirements-jetson.txt
python -c "import cryptography; print('cryptography OK', cryptography.__version__)"

rm -rf chroma_db
python -m rag.indexer 2>&1 | tee index_run.log

grep -i "SKIP" index_run.log
```
