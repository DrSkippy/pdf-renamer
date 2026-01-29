def clean_text(raw_text_from_pdf):
    """
    Cleans text by filtering out lines that are too short.

    Processes multi-line text by splitting it into individual lines, stripping
    whitespace, and removing lines that don't meet the minimum character length
    threshold. Lines shorter than MIN_LINE_CHAR_THRESHOLD are excluded to filter
    out titles, authors, dates, or other short content that may not be relevant
    for further processing.

    :param raw_text_from_pdf: Multi-line text string to be cleaned
    :type raw_text_from_pdf: str
    :return: List of cleaned text lines that meet the minimum length requirement
    :rtype: list
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
    logging.info("Starting extraction...")
    raw_text_fragment_from_pdf = clean_text(raw_text_fragment_from_pdf)
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

