import logging
from dateparser.search import search_dates
from pathlib import Path
from pypdf import PdfReader
import re
import json
from llms.extractors import summarize_text, llm_title, llm_authors

MIN_LINE_CHAR_THRESHOLD = 5  # min title/author/date line size
MIN_CONTENT_CHARS = 5000   # min characters to be considered the first page of the content
MAX_LINES_FOR_TITLE_AND_AUTHORS = 16

def clean_text(raw_text_from_pdf):
    """
    Clean and filter text extracted from PDF by removing short lines that likely
    contain metadata rather than content.

    This function processes raw PDF text by splitting it into lines, trimming
    whitespace, and filtering out lines that fall below a minimum character
    threshold. Lines shorter than the threshold are considered to be metadata
    such as titles, authors, or dates rather than actual content.

    :param raw_text_from_pdf: The raw text string extracted from a PDF document
    :type raw_text_from_pdf: str
    :return: List of cleaned text lines that meet the minimum character threshold,
             with whitespace trimmed from each line
    :rtype: list[str]
    """
    logging.info("Cleaning text...")
    result = []
    for row in raw_text_from_pdf.split("\n"):
        tmp = row.strip()
        if len(tmp) > MIN_LINE_CHAR_THRESHOLD:
            # No titles, authors or date smaller than threshold
            result.append(tmp)
        else:
            logging.info(f"Skipping line -- too short {len(tmp)} < {MIN_LINE_CHAR_THRESHOLD} chars")
    return result


def likely_title(raw_text_fragment_from_pdf):
    """
    Extracts likely title, authors, and date from raw text fragments of a PDF.

    This function processes the initial lines of text from a PDF document to
    identify and extract metadata including the document title, authors, and
    publication date. It uses a combination of pattern matching for dates and
    LLM-based extraction for titles and authors. The function accumulates text
    lines up to a maximum threshold and delegates title and author extraction
    to specialized LLM functions.

    :param raw_text_fragment_from_pdf: Raw text lines extracted from the
        beginning of a PDF document
    :type raw_text_fragment_from_pdf: list
    :return: A tuple containing the extracted title (str or None), authors
        (or None), and date information as a dictionary with 'date' and
        'date_line' keys (dict or None)
    :rtype: tuple
    """
    logging.info("Starting extraction...")
    title, authors, date = None, None, None
    title_accum = []
    for text_line in raw_text_fragment_from_pdf[:MAX_LINES_FOR_TITLE_AND_AUTHORS]:
        # Assumptions are we have titles and authors within 8 lines...
        if date is None:
            # dates can be part of title, so continue with other options below
            dates = search_dates(text_line.strip())
            if dates is not None and len(dates) > 0:
                date = {"date": str(dates[0]), "date_line": text_line.strip() }
        else:
            title_accum.append(text_line)
    title = llm_title(title_accum)
    authors = llm_authors(title_accum)
    return title, authors, date

def extract_from_pdf(pdf_path: Path):
    """
    Extracts metadata and summary from a PDF file by processing its pages.

    This function reads a PDF file and attempts to extract key information including
    the title, authors, publication date, and a text summary. It ensures sufficient
    content is extracted by reading multiple pages if the first page contains too
    little text. The extracted text is cleaned and summarized before returning the
    results.

    :param pdf_path: File system path to the PDF document to be processed
    :type pdf_path: Path
    :return: A tuple containing the extracted title, authors list, publication date,
             and text summary from the PDF document
    :rtype: tuple
    """
    logging.info(f"Extracting from pdf {pdf_path}...")
    pdf_text = []
    reader = PdfReader(str(pdf_path))

    page_text = reader.pages[0].extract_text()
    pdf_text.extend(clean_text(page_text))

    page_index = 1
    while len(page_text) < MIN_CONTENT_CHARS:
        logging.warning(f"First page text too short ({len(first_page_text)} chars), trying another page")
        page_text = reader.pages[page_text].extract_text()
        pdf_text.extend(clean_text(page_text))
        page_index += 1

    cont_pages_text = "\n".join(pages_text)
    summary = summarize_text(cont_pages_text)
    title, authors, date = likely_title(page_text)
    return title , authors, date, summary

