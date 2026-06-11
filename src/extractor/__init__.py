"""Extractor package — document text extraction and indexing pipeline.

Scripts in this package form the upstream pipeline for personal-rag:
  extract_text      → PDF/EPUB/MD text extraction
  enrich_metadata   → metadata enrichment from embedded fields + ISBNs
  build_index_documents → join inventory + extracted text into index JSONs
  build_obsidian_notes  → generate Obsidian resource note stubs
  build_sqlite      → build FTS5 SQLite database from JSONL
  build_vault_index → index vault Knowledge/ notes into JSONL
  dup_detect        → near-duplicate detection by ISBN + shingle similarity
  link_mocs         → inject managed resource backlinks into Topic MOCs
  search            → CLI full-text search over resources.db
  analyze_files     → pre-flight survey of a document directory
"""
