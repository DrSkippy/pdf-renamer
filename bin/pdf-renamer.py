import json
import logging
import ollama
import re
import tqdm
from dateparser.search import search_dates
from pathlib import Path

from utils.file_name import make_filename_safe
from utils.pdf_content import extract_from_pdf

FORMAT = "[%(asctime)s | %(name)s | %(levelname)s | %(filename)s:%(funcName)s():%(lineno)d] %(message)s"
logging.basicConfig(format=FORMAT, filename="process.log", level=logging.DEBUG)

pdf_root_path = Path("/home/scott/ownCloud/Documents/Articles and Papers/")

logging.info("Starting processing")
logging.info(f"Reading pdfs from {pdf_root_path}")
if __name__ == "__main__":
    for i, filename in enumerate(tqdm.tqdm(list(pdf_root_path.glob("*.pdf")))):
        logging.info(f"Processing file {filename}")
        title, authors, date = extract_from_pdf(filename)
        clean_filename = make_filename_safe(title["title"]) + ".json"
        record = {"title": title, "authors": authors, "date": date, "summary": None, "source": str(filename),
                  "destination": clean_filename}
        maxp = min([len(reader.pages), 8])
        pages_text = [reader.pages[j].extract_text() for j in range(maxp)]

        with open("./output/" + clean_filename, "w") as f:
            json.dump(record, f)

    print(f"Total Documents: {i}")
