def json_stringify(x):
    logging.info(f"Stringifying json: {x}")
    x = x.strip()
    x = x.strip("`")
    if x.startswith("json"):
        x = x[4:]
    x = x.replace('\\"',"'")
    x = x.replace("\\", "\\\\")
    logging.info(f"Stringified json: {x}")
    return json.loads(x)

def summarize_text(full_text):
    logging.info(f"Summarizing with model {SUMMARY_MODEL}...")
    response = client.chat(
        model=SUMMARY_MODEL,
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant that extracts information from text. The user will provide you with a text document, and your task is to create a 1-2 paragraph abstract. Format the result as json.",
            },
            {"role": "user", "content": full_text},
        ],
    )
    return json_stringify(response["message"]["content"])


def llm_authors(x):
    logging.info(f"Getting authors with model {AUTHORS_MODEL}...")
    full_text = "\n".join(x)
    response = client.chat(
        model=AUTHORS_MODEL,
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant that extracts information from text. The user will provide you with a text document, and your task is to extract the most likely line containing authors names. Find the most like line containing the authors name. Reaturn the result in json with the line number, the line of text selected and an array of the authors names.",
            },
            {"role": "user", "content": full_text},
        ],
    )
    return json_stringify(response["message"]["content"])


def llm_title(x):
    logging.info(f"Getting title with model {TITLE_MODEL}...")
    full_text = " ".join(x)
    response = client.chat(
        model=TITLE_MODEL,
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant that extracts information from text. The user will provide you with a text document that likely contains the title of the document in the first few lines of text. Your task is to extract the most likely line or lines containing the full title of the paper. Find the most like line or lines containing the title. Return the result in json with the line number of the first line of the title, and the full title fo the document.",
            },
            {"role": "user", "content": full_text},
        ],
    )
    return json_stringify(response["message"]["content"])


