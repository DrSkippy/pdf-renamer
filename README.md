# pdf-renamer

Rename PDF files using their actual document title, extracted from content by an LLM via [OpenRouter](https://openrouter.ai/) (`openrouter/auto`), or optionally selected from the text with [TypeSafe](https://docs.typesafe.ai/). Supports a dry-run / review / apply workflow so renames can be inspected and edited before any files are touched.

## How it works

```
PDF file
  └─ PyPDF: extract_text()
       └─ if page is image-based → OCR via openrouter/auto
  └─ clean_text(): filter lines < 2 chars
  └─ likely_title(): send first 30 lines to openrouter/auto
       │   (--extractor typesafe: title and date selected from candidates instead)
       ├─ llm_title()   → {"title": "..."}
       └─ llm_authors() → {"authors": "...", "authors_list": [...]}
  └─ summarize_text(): send first 4000 chars to openrouter/auto
       └─ {"summary": "..."}
  └─ make_filename_safe() → filesystem-safe stem
  └─ write metadata JSON or rename file
```

### LLM model assignments

Every task uses OpenRouter's auto router (`openrouter/auto`), which picks a model per
prompt. Each task has its own constant in `llms/extractors.py` (`TITLE_MODEL`,
`AUTHORS_MODEL`, `SUMMARY_MODEL`, `OCR_MODEL`) so any one can be pinned to a specific
OpenRouter model. JSON tasks request a strict JSON-schema response and set
`provider.require_parameters` so routing only reaches providers that support it.

| Task | Model |
|------|-------|
| Title extraction | `openrouter/auto` (or TypeSafe `jev-latest` with `--extractor typesafe`) |
| Author extraction | `openrouter/auto` |
| Summarization | `openrouter/auto` |
| OCR fallback | `openrouter/auto` |

### Text size limits

| Constant | Value | Controls |
|----------|-------|---------|
| `MAX_LINES_FOR_TITLE_AND_AUTHORS` | 30 lines | Input to title/author LLM calls |
| `MAX_SUMMARY_CHARS` | 4000 chars | Input to summarization LLM call |
| `MAX_PAGES_TO_READ` | 3 pages | Max PDF pages read before stopping |
| `MIN_CONTENT_LINES` | 528 lines | Target line count that triggers reading an extra page |
| `MIN_OCR_TRIGGER_CHARS` | 50 chars | PyPDF output below this triggers OCR fallback |

## Requirements

- Python 3.11+
- [Poetry](https://python-poetry.org/)
- An [OpenRouter](https://openrouter.ai/) API key
- A [TypeSafe](https://docs.typesafe.ai/) API key (only for `--extractor typesafe`)

Credentials are read from the environment, loaded from `.envrc` by
[direnv](https://direnv.net/). Copy `.envrc.example` to `.envrc`, fill in the keys,
and run `direnv allow`.

## Setup

```bash
poetry install
```

## Usage

### Direct rename (default)

Runs LLM extraction and renames each PDF in place immediately.

```bash
poetry run python bin/pdf-renamer.py --pdf-root /path/to/pdfs/
```

Add `--json PATH` to also write a metadata JSON file per PDF to a directory:

```bash
poetry run python bin/pdf-renamer.py --pdf-root /path/to/pdfs/ --json ./output/
```

Add `--extractor typesafe` to select the title and publication date from candidate
spans in the text with TypeSafe instead of having the LLM write them. The title is
copied exactly from the PDF; low-confidence picks fall back to the original filename.
Authors and summary still use OpenRouter.

```bash
poetry run python bin/pdf-renamer.py --dry-run --extractor typesafe --pdf-root /path/to/pdfs/
```

### Recommended for large collections: dry-run → review → apply

```bash
# Step 1: run LLM extraction, preview proposed renames, save plan
poetry run python bin/pdf-renamer.py --dry-run --pdf-root /path/to/pdfs/

# Output:
#   original_filename.pdf  →  Clean_Document_Title.pdf
#   ...
#   Plan saved to ./rename_plan.json  (42 files)

# Step 2: review (and optionally edit) rename_plan.json

# Step 3: apply the renames — no LLM calls, no re-processing
poetry run python bin/pdf-renamer.py --apply
```

### All options

```
--pdf-root PATH       Directory of PDF files to process
                      (default: ~/ownCloud/Documents/Articles and Papers/)
--plan-file PATH      Rename plan JSON for --dry-run / --apply
                      (default: ./rename_plan.json)
--json PATH           Write one metadata JSON file per PDF to this directory
                      (created if absent; use with default rename mode)
--log-path PATH       Log file location (default: process.log)
--log-level LEVEL     DEBUG | INFO | WARNING | ERROR | CRITICAL (default: DEBUG)
--dry-run             Run extraction, print proposed renames, save plan file
--apply               Read plan file and perform renames (mutually exclusive with --dry-run)
```

### Rename plan format

`rename_plan.json` is a JSON array. Each entry can be edited before `--apply`:

```json
[
  {
    "source": "/path/to/pdfs/messy_name_2024.pdf",
    "destination": "/path/to/pdfs/A_Tutorial_on_Spectral_Clustering.pdf",
    "title": {"title": "A Tutorial on Spectral Clustering"},
    "authors": {"authors": "Ulrike von Luxburg", "authors_list": ["Ulrike von Luxburg"]},
    "date": null,
    "summary": {"summary": "This paper presents..."}
  }
]
```

`--apply` skips entries where `source` no longer exists or `destination` already exists.

## Project structure

```
pdf-renamer/
├── bin/
│   └── pdf-renamer.py      CLI entry point (dry-run / apply / full modes)
├── llms/
│   └── extractors.py       OpenRouter client; title, author, summary, and OCR extraction
│   └── typesafe_selectors.py  TypeSafe title/date selection (--extractor typesafe)
├── utils/
│   ├── pdf_content.py      PDF reading pipeline, OCR fallback, text limits
│   └── file_name.py        Filesystem-safe filename sanitization
├── tests/
│   ├── test_extractors.py  Unit tests for OpenRouterExtractors
│   ├── test_pdf_content.py Unit tests for PDF processing pipeline
│   └── test_integration.py Integration tests against sample PDFs (require OpenRouter key)
├── samples/                Sample PDFs used by integration tests
├── pyproject.toml
└── poetry.lock
```

## Running tests

```bash
# Unit tests (no API keys required)
poetry run pytest --cov=llms --cov=utils --cov-report=term-missing tests/test_extractors.py tests/test_pdf_content.py

# Integration tests (require OPENROUTER_API_KEY)
poetry run pytest -m integration tests/test_integration.py -v
```
