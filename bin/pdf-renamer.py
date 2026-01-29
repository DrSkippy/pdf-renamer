import logging
import json
import tqdm
from pathlib import Path

from utils.file_name import make_filename_safe
from utils.pdf_content import extract_from_pdf

PDF_ROOT_PATH = "/home/scott/ownCloud/Documents/Articles and Papers/"
FORMAT = "[%(asctime)s | %(name)s | %(levelname)s | %(filename)s:%(funcName)s():%(lineno)d] %(message)s"
logging.basicConfig(format=FORMAT, filename="process.log", level=logging.DEBUG)

if __name__ == "__main__":
    pdf_root_path = Path(PDF_ROOT_PATH)
    logging.info("Starting processing")
    logging.info(f"Reading pdfs from {pdf_root_path}")
    for i, filename in enumerate(tqdm.tqdm(list(pdf_root_path.glob("*.pdf")))):
        logging.info(f"Processing file {filename}")
        title, authors, date, summary = extract_from_pdf(filename)
        if title["title"] is None or len(title["title"]) == 0:
            logging.info("Falling back to file title.")
            title["title"] = make_filename_safe(filename[:-4])
        clean_filename_stem = make_filename_safe(title["title"])
        record = {"title": title,
                  "authors": authors,
                  "date": date,
                  "summary": summary,
                  "source": str(filename),
                  "destination": clean_filename_stem + ".pdf"}

        with open("./output/" + clean_filename_stem + ".json", "w") as f:
            json.dump(record, f)

    print(f"Total Documents: {i}")
