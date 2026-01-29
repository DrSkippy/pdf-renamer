import json
import logging
import ollama
from pydantic import BaseModel, ValidationError


class Title(BaseModel):
    title: str
    line_number: int


class Authors(BaseModel):
    authors_list: list[str]
    authors: str
    line_number: int


class Summary(BaseModel):
    summary: str


class OllamaExtractors:
    TITLE_MODEL = "gpt-oss:latest"
    TITLE_MODEL_PROMPT = ("You are a helpful assistant that extracts specific information from text. The user will "
                          "provide you with a text document that contains the title of the document "
                          "in the first few lines of text. Your task is to extract the most likely line or "
                          "lines containing the full title of the document. Return the result in json with the "
                          " keys 'line_number' indicating the first line of the 'title', title for the full title "
                          "of the document.")
    SUMMARY_MODEL = "gpt-oss:latest"
    SUMMARY_MODEL_PROMPT = ("You are a helpful assistant that extracts information from text. The user will "
                            "provide you with a text document, and your task is to create a 1-2 paragraph "
                            "abstract. Format the result as json with key 'summary'.")
    AUTHORS_MODEL = "gpt-oss:latest"
    AUTHORS_MODEL_PROMPT = ("You are a helpful assistant that extracts specific information from text. The user will "
                            "provide you with a text document, and your task is to extract the author(s) of "
                            "the document. Find the most likely line or lines "
                            "containing the article authors. Return the result in json with keys 'line_number' of "
                            "the first line of the authors, 'authors' for the the authors string and "
                            "'authors_list' with a list of each author.")
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
        return x

    def summarize_text(self, full_text):
        logging.info(f"Summarizing with model {self.SUMMARY_MODEL}...")
        response = self.client.chat(
            model=self.SUMMARY_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": self.SUMMARY_MODEL_PROMPT
                },
                {"role": "user", "content": full_text},
            ],
        )
        try:
            t = Summary.model_validate_json(self.json_loads_with_stringify(response["message"]["content"]))
        except ValidationError as e:
            logging.error(f"Failed to synthesize summary from ollama response: {e}")
            logging.error(f"Failed to synthesize summary from ollama response: {response['message']['content']}")
            t = Summary(summary='')
        result = t.model_dump(mode='json')
        return result

    def llm_authors(self, x):
        logging.info(f"Getting authors with model {self.AUTHORS_MODEL}...")
        full_text = "\n".join(x)
        response = self.client.chat(
            model=self.AUTHORS_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": self.AUTHORS_MODEL_PROMPT
                },
                {"role": "user", "content": full_text},
            ],
        )
        try:
            t = Authors.model_validate_json(self.json_loads_with_stringify(response["message"]["content"]))
        except ValidationError as e:
            logging.error(f"Failed to synthesize authors from ollama response: {e}")
            logging.error(f"Failed to parse authors from ollama response: {response['message']['content']}")
            t = Authors(authors_list=[], authors='', line_number=0)
        result = t.model_dump(mode='json')
        return result

    def llm_title(self, x):
        logging.info(f"Getting title with model {self.TITLE_MODEL}...")
        full_text = " ".join(x)
        response = self.client.chat(
            model=self.TITLE_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": self.TITLE_MODEL_PROMPT
                },
                {"role": "user", "content": full_text},
            ],
        )
        try:
            t = Title.model_validate_json(self.json_loads_with_stringify(response["message"]["content"]))
        except ValidationError as e:
            logging.error(f"Failed to synthesize title from ollama response: {e}")
            logging.error(f"Failed to parse title from ollama response: {response['message']['content']}")
            t = Title(title='', line_number=0)
        result = t.model_dump(mode='json')
        return result
