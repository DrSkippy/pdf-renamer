# pdf-renamer

Rename PDF files using title and author data extracted from the document content. Uses a local Ollama server for LLM-based title, author, and summary extraction.

## Overview

This utility processes a directory of PDF files, extracts metadata (title, authors, date, summary) using an LLM, and generates clean, descriptive filenames. The workflow is split into two steps:

1. **Extract metadata** - Scan PDFs and create JSON mapping files
2. **Copy with new names** - Copy original files to output directory with renamed filenames

## Requirements

- Python 3.x
- Poetry for dependency management
- Local Ollama server running

## Installation

```bash
poetry install
```

## Usage

### Step 1: Extract Metadata and Generate Rename Mappings

Edit `bin/pdf-renamer.py` to set your PDF source directory:

```python
PDF_ROOT_PATH = "/path/to/your/pdf/directory/"
```

Run the extraction:

```bash
poetry run python bin/pdf-renamer.py
```

This scans all PDFs in the source directory and creates JSON files in `./output/` containing:
- `title` - Extracted document title
- `authors` - Extracted author information
- `date` - Publication date (if found)
- `summary` - LLM-generated summary
- `source` - Original file path
- `destination` - Suggested new filename

### Step 2: Copy Files with New Names

After reviewing the JSON mappings in `./output/`, run:

```bash
poetry run python bin/copy-renamed.py
```

This reads each JSON file and copies the source PDF to `./output/` with the new destination filename.

## Output

The `./output/` directory will contain:
- JSON metadata files for each processed PDF
- Renamed copies of the original PDF files

## How It Works

1. **PDF Text Extraction** - Uses pypdf to extract text from PDF pages
2. **Date Detection** - Uses dateparser to find publication dates in the text
3. **LLM Extraction** - Sends text to Ollama for title, author, and summary extraction
4. **Filename Sanitization** - Converts extracted titles to safe filenames (replaces special characters, limits length)
