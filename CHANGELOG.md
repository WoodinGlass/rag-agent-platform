# Changelog

All notable changes to this project will be documented here.
Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Versioning: [SemVer](https://semver.org/).

## [Unreleased]

### Added
- **M1 — Local RAG (deterministic core)**
  - `app/core`: pydantic-settings config, deterministic ids, JSON logging
  - `app/rag/chunker.py`: recursive chunker, version-tagged, deterministic
  - `app/rag/embedder.py`: provider-agnostic interface + `FakeEmbedder`
  - `app/rag/store.py`: `VectorStore` ABC, `MemoryStore`, lazy `ChromaStore`
  - `app/rag/pipeline.py`: idempotent ingest (`sha256 + chunker_version`)
  - `evals/run.py`: retrieval hit-rate@k CLI
  - `scripts/ingest_demo.py`: offline ingest demo
  - Unit tests (ids, chunker, embedder, pipeline) + integration test (Chroma)
- CI workflow: lint (ruff) + unit tests (pytest) on push/PR
- Initial repo scaffold: README, LICENSE, pyproject, .env.example, .gitignore

### Verified
- `pytest -m "not integration"` green
- Retrieval hit-rate@5 = **1.000** on demo set (12/12)
- Idempotent ingest: re-running `ingest_demo.py` reports `skipped=12`

## [0.1.0] - 2025-XX-XX

### Added
- Initial scaffold
