for i, filename in enumerate(tqdm.tqdm(list(pdf_root_path.glob("*.pdf")))):
    logging.info(f"Processing file {filename}")
    logging.info(f"Reading file number {i}...")
    reader = PdfReader(filename)

    page_index = 0
    title = None
    while title is None:
        page = reader.pages[page_index]
        page_text = page.extract_text()
        if len(page_text) < MIN_TITLE_PAGE_CHARS:
            page_index += 1
            continue
        title, authors, date = likely_title(page_text)

    clean_filename = make_filename_safe(title["title"]) + ".json"
    record = {"title": title, "authors": authors, "date": date, "summary": None, "source": str(filename), "destination": clean_filename}
    maxp = min([len(reader.pages), 8])
    pages_text = [reader.pages[j].extract_text() for j in range(maxp)]
    pages_text = "\n".join(pages_text)
    record["summary"] = summarize_text(pages_text)

    with open("./output/" + clean_filename, "w") as f:
        json.dump(record, f)

print(f"Total Documents: {i}")

