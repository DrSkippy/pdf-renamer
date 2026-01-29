import logging
import ollama
from dateparser.search import search_dates
from pathlib import Path
from pypdf import PdfReader
import tqdm
import re
import json

FORMAT = "[%(asctime)s | %(name)s | %(levelname)s | %(filename)s:%(funcName)s():%(lineno)d] %(message)s"
logging.basicConfig(format=FORMAT, filename="process.log", level=logging.DEBUG)
HOST="http://lambda-dual.home.lan:11434"
MIN_LINE_CHAR_THRESHOLD = 5  # min title/author/date line size
MIN_TITLE_PAGE_CHARS = 25 # min characters to be considered the first page of the content
TITLE_MODEL = "gpt-oss:latest"
SUMMARY_MODEL = "gpt-oss:latest"
AUTHORS_MODEL = "gpt-oss:latest"
MAX_LINES_FOR_TITLE_AND_AUTHORS = 11

pdf_root_path = Path("/home/scott/ownCloud/Documents/Articles and Papers/")
logging.info("Starting processing")
logging.info(f"Reading pdfs from {pdf_root_path}")
client = ollama.Client(host=HOST)
logging.info(f"Using ollama client against host at {HOST}")


def make_filename_safe(filename):
    # Replace all spaces with a single underscore first
    filename = re.sub(r"\s+", "_", filename)
    # Remove any characters that are not ASCII alphanumeric, underscores, or hyphens
    filename = re.sub(r"[^a-zA-Z0-9_.-]", "", filename)
    # Replace multiple consecutive underscores with a single underscore
    filename = re.sub(r"__+", "_", filename)
    # Remove leading/trailing underscores
    filename = filename.strip("_")
    return filename

