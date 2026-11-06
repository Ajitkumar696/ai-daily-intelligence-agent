# Safe Cleanup Audit

Audited source references, CLI entry points, source configuration, tests,
requirements, project documentation, generated output, and repository status.
No uncertain files were deleted.

## SAFE TO REMOVE

None. No project file was confirmed to be an unused duplicate or disposable
temporary artifact.

## REVIEW REQUIRED

- `src/summarization/providers.py` still contains the legacy Groq client and
  settings. The application factory no longer selects Groq, but provider tests
  still exercise the class directly. Remove only after deciding whether this
  compatibility surface and its tests should be retired.

## KEEP

- All configured collectors and source configuration, including optional
  authenticated API collectors and the existing Crawl4AI adapter.
- CLI entry points, research and summarization modules, PDF/video modules,
  schemas, URL normalization, citation validation, deduplication, and history.
- Tests, README, requirements files, `.env.example`, and `.env`.
- Existing `output/` JSON files, processed history, summaries/cache, PDFs,
  videos, and other generated research artifacts.
- `.gitignore`, `pytest.ini`, and existing Git metadata.

No generated outputs, caches, project secrets, tests, documentation, or
workflows were deleted or overwritten. No `.github/` workflow files were
present in the inspected project tree.
