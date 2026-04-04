# pdf-renamer

Rename PDF files using their actual document title, extracted from content by a local [Ollama](https://ollama.com/) LLM. Supports a dry-run / review / apply workflow so renames can be inspected and edited before any files are touched.

## How it works

```
PDF file
  └─ PyPDF: extract_text()
       └─ if page is image-based → OCR via deepseek-ocr (Ollama)
  └─ clean_text(): filter lines < 2 chars
  └─ likely_title(): send first 30 lines to qwen3.5 (Ollama)
       ├─ llm_title()   → {"title": "..."}
       └─ llm_authors() → {"authors": "...", "authors_list": [...]}
  └─ summarize_text(): send first 4000 chars to gpt-oss (Ollama)
       └─ {"summary": "..."}
  └─ make_filename_safe() → filesystem-safe stem
  └─ write metadata JSON or rename file
```

### LLM model assignments

| Task | Model | Reason |
|------|-------|--------|
| Title extraction | `qwen3.5:latest` | Lightweight general-purpose; handles multilingual academic text |
| Author extraction | `qwen3.5:latest` | Same |
| Summarization | `gpt-oss:latest` | Larger model suited to longer-form generation |
| OCR fallback | `deepseek-ocr:latest` | Purpose-built for image/scanned document text extraction |

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
- A running [Ollama](https://ollama.com/) instance with the following models pulled:
  - `qwen3.5:latest`
  - `gpt-oss:latest`
  - `deepseek-ocr:latest`

The Ollama host is configured in `llms/extractors.py` (`HOST = "http://192.168.1.90:11434"`).

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
│   └── extractors.py       Ollama client; title, author, summary, and OCR extraction
├── utils/
│   ├── pdf_content.py      PDF reading pipeline, OCR fallback, text limits
│   └── file_name.py        Filesystem-safe filename sanitization
├── tests/
│   ├── test_extractors.py  Unit tests for OllamaExtractors
│   ├── test_pdf_content.py Unit tests for PDF processing pipeline
│   └── test_integration.py Integration tests against sample PDFs (require live Ollama)
├── samples/                Sample PDFs used by integration tests
├── pyproject.toml
└── poetry.lock
```

## Running tests

```bash
# Unit tests (no Ollama required)
poetry run pytest --cov=llms --cov=utils --cov-report=term-missing tests/test_extractors.py tests/test_pdf_content.py

# Integration tests (require live Ollama with models pulled)
poetry run pytest -m integration tests/test_integration.py -v
```
