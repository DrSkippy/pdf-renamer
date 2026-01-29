import json
import logging
import ollama


class OllamaExtractors:
    TITLE_MODEL = "gpt-oss:latest"
    TITLE_MODEL_PROMPT = ("You are a helpful assistant that extracts information from text. The user will "
                          "provide you with a text document that likely contains the title of the document "
                          "in the first few lines of text. Your task is to extract the most likely line or "
                          "lines containing the full title of the paper. Find the most like line or lines "
                          "containing the title. Return the result in json with the line number of the first "
                          "line of the title, and the full title fo the document.")
    SUMMARY_MODEL = "gpt-oss:latest"
    SUMMARY_MODEL_PROMPT = ("You are a helpful assistant that extracts information from text. The user will "
                            "provide you with a text document, and your task is to create a 1-2 paragraph "
                            "abstract. Format the result as json.")
    AUTHORS_MODEL = "gpt-oss:latest"
    AUTHORS_MODEL_PROMPT = ("You are a helpful assistant that extracts information from text. The user will "
                            "provide you with a text document, and your task is to extract the author(s) of "
                            "the document. Format the result as json.")
    HOST = "http://lambda-dual.home.lan:11434"

    def __init__(self):
        self.client = ollama.Client(host=self.HOST)
        logging.info(f"Using ollama client against host at {self.HOST}")

    def json_loads_with_stringify(self, x):
        logging.debug(f"Stringifying json: {x}")
        x = x.strip()
        x = x.strip("`")
        if x.startswith("json"):
            x = x[4:]
        x = x.replace('\\"', "'")
        x = x.replace("\\", "\\\\")
        logging.info(f"Stringified json: {x}")
        return json.loads(x)

    def summarize_text(self, full_text):
        logging.info(f"Summarizing with model {self.SUMMARY_MODEL}...")
        response = self.client.chat(
            model=self.SUMMARY_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": SUMMARY_MODEL_PROMPT
                },
                {"role": "user", "content": full_text},
            ],
        )
        return self.json_loads_with_stringify(response["message"]["content"])

    def llm_authors(self, x):
        logging.info(f"Getting authors with model {self.AUTHORS_MODEL}...")
        full_text = "\n".join(x)
        response = self.client.chat(
            model=self.AUTHORS_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": AUTHORS_MODEL_PROMPT
                },
                {"role": "user", "content": full_text},
            ],
        )
        return self.json_loads_with_stringify(response["message"]["content"])

    def llm_title(self, x):
        logging.info(f"Getting title with model {self.TITLE_MODEL}...")
        full_text = " ".join(x)
        response = self.client.chat(
            model=self.TITLE_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": TITLE_MODEL_PROMPT
                },
                {"role": "user", "content": full_text},
            ],
        )
        return self.json_loads_with_stringify(response["message"]["content"])
