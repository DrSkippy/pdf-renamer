import logging
from dateparser.search import search_dates
from pathlib import Path
from pypdf import PdfReader
from llms.extractors import OllamaExtractors

MIN_LINE_CHAR_THRESHOLD = 2    # min chars for a line to be kept
MIN_CONTENT_LINES = 66 * 8    # target line count before stopping page reads
MAX_PAGES_TO_READ = 3         # hard cap on pages read regardless of line count
MAX_LINES_FOR_TITLE_AND_AUTHORS = 30
MAX_SUMMARY_CHARS = 4000      # max chars sent to the summary LLM (~1k tokens)
MIN_OCR_TRIGGER_CHARS = 50    # if PyPDF extracts fewer chars from a page, try OCR


def clean_text(raw_text_from_pdf: str) -> list[str]:
    """Clean and filter text extracted from PDF.

    Splits raw PDF text into lines, strips whitespace, and filters out lines
    below the minimum character threshold.

    :param raw_text_from_pdf: The raw text string extracted from a PDF document
    :type raw_text_from_pdf: str
    :return: List of cleaned text lines meeting the minimum character threshold
    :rtype: list[str]
    """
    logging.info("Cleaning text...")
    result = []
    for row in raw_text_from_pdf.split("\n"):
        tmp = row.strip()
        if len(tmp) > MIN_LINE_CHAR_THRESHOLD:
            result.append(tmp)
        else:
            logging.info(f"Skipping line-too short {len(tmp)} < {MIN_LINE_CHAR_THRESHOLD} chars")
    return result


def _extract_page_text(page: object, extractor: OllamaExtractors) -> str:
    """Extract text from a single PDF page, falling back to OCR if needed.

    Tries PyPDF text extraction first. If the result is below MIN_OCR_TRIGGER_CHARS
    and the page contains embedded images, delegates to the OCR model.

    :param page: A pypdf PageObject
    :param extractor: Configured OllamaExtractors instance
    :return: Extracted text string (may be empty if all methods fail)
    :rtype: str
    """
    text = page.extract_text() or ""
    if len(text.strip()) < MIN_OCR_TRIGGER_CHARS:
        page_images = list(page.images)
        if page_images:
            logging.warning(
                f"Page has minimal extracted text ({len(text.strip())} chars); "
                "attempting OCR fallback..."
            )
            text = extractor.ocr_page_images(page_images)
    return text


def likely_title(
    raw_text_fragment_from_pdf: list[str], extractor: OllamaExtractors
) -> tuple:
    """Extract title, authors, and date from the opening lines of a PDF.

    Sends the first MAX_LINES_FOR_TITLE_AND_AUTHORS lines to the LLM for title
    and author extraction. Date detection runs as an independent scan over the
    same lines and does not affect which lines are sent to the LLM.

    :param raw_text_fragment_from_pdf: Cleaned text lines from the PDF
    :type raw_text_fragment_from_pdf: list[str]
    :param extractor: Configured OllamaExtractors instance
    :type extractor: OllamaExtractors
    :return: Tuple of (title_dict, authors_dict, date_dict or None)
    :rtype: tuple
    """
    logging.info("Starting extraction...")
    date = None
    title_lines = list(raw_text_fragment_from_pdf[:MAX_LINES_FOR_TITLE_AND_AUTHORS])

    # Date scan is independent — does not gate which lines go to the LLM
    for text_line in title_lines:
        dates = search_dates(text_line.strip())
        if dates is not None and len(dates) > 0:
            date = {"date": str(dates[0]), "date_line": text_line.strip()}
            break

    title = extractor.llm_title(title_lines)
    authors = extractor.llm_authors(title_lines)
    return title, authors, date


def extract_from_pdf(pdf_path: Path) -> tuple:
    """Extract metadata and summary from a PDF file.

    Reads the PDF, cleans the text, and calls LLMs to extract title, authors,
    date, and a summary. Reads additional pages if the first page has too little
    content. Falls back to OCR for image-based pages.

    :param pdf_path: Path to the PDF file
    :type pdf_path: Path
    :return: Tuple of (title_dict, authors_dict, date_dict or None, summary_dict)
    :rtype: tuple
    """
    logging.info(f"Extracting from pdf {pdf_path}...")
    pdf_text: list[str] = []
    reader = PdfReader(str(pdf_path))
    extractor = OllamaExtractors()

    page_text = _extract_page_text(reader.pages[0], extractor)
    pdf_text.extend(clean_text(page_text))

    page_index = 1
    while (
        len(pdf_text) < MIN_CONTENT_LINES
        and page_index < len(reader.pages)
        and page_index < MAX_PAGES_TO_READ
    ):
        logging.warning(
            f"First {page_index + 1} page(s) of text too short ({len(pdf_text)} lines), adding page"
        )
        page_text = _extract_page_text(reader.pages[page_index], extractor)
        pdf_text.extend(clean_text(page_text))
        page_index += 1

    cont_pdf_text = "\n".join(pdf_text)[:MAX_SUMMARY_CHARS]
    summary = extractor.summarize_text(cont_pdf_text)
    title, authors, date = likely_title(pdf_text, extractor)
    return title, authors, date, summary
