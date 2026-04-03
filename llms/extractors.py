import logging
import re
import ollama
from pydantic import BaseModel, ValidationError


class Title(BaseModel):
    title: str


class Authors(BaseModel):
    authors_list: list[str]
    authors: str


class Summary(BaseModel):
    summary: str


class OllamaExtractors:
    # Text analysis tasks: structured extraction from already-decoded text
    TITLE_MODEL = "qwen3.5:latest"
    TITLE_MODEL_PROMPT = (
        "You are extracting metadata from the first page of an academic paper or document. "
        "The text below contains the beginning of the document, with one line per input line. "
        "The title is typically the largest or most prominent text at the top, before authors, "
        "affiliations, abstract, or publication details. "
        "Return JSON with a single key 'title' containing the full document title as a string. "
        "If the title spans multiple lines, join them into one string. "
        "Do not include journal names, authors, or subtitles unless clearly part of the main title. "
        'Example output: {"title": "Deep Learning for Natural Language Processing"}'
    )
    AUTHORS_MODEL = "qwen3.5:latest"
    AUTHORS_MODEL_PROMPT = (
        "You are extracting metadata from the first page of an academic paper or document. "
        "The text below contains the beginning of the document, with one line per input line. "
        "Authors typically appear directly below the title, before the abstract. "
        "Return JSON with keys: "
        "'authors': a single string with all author names (comma-separated), "
        "'authors_list': a list of individual author name strings. "
        'If no authors are found, return {"authors": "", "authors_list": []}. '
        'Example: {"authors": "Jane Smith, John Doe", "authors_list": ["Jane Smith", "John Doe"]}'
    )
    # Summarization: longer-form generation benefits from the larger model
    SUMMARY_MODEL = "gpt-oss:latest"
    SUMMARY_MODEL_PROMPT = (
        "You are a helpful assistant that extracts information from text. The user will "
        "provide you with a text document, and your task is to create a 1-2 paragraph "
        "abstract. Format the result as json with key 'summary'."
    )
    # OCR fallback: used only when PyPDF cannot extract text (scanned/image-based PDFs)
    OCR_MODEL = "deepseek-ocr:latest"
    OCR_MODEL_PROMPT = (
        "Extract all text from this image exactly as it appears. "
        "Return only the raw extracted text with no commentary or formatting."
    )
    HOST = "http://192.168.1.90:11434"

    def __init__(self) -> None:
        self.client = ollama.Client(host=self.HOST)
        logging.info(f"Using ollama client against host at {self.HOST}")

    def json_loads_with_stringify(self, x: str) -> str:
        """Extract a JSON object string from an LLM response.

        Handles markdown code fences, prose wrappers, and other common LLM
        response formats. Returns the raw JSON string for Pydantic to parse.
        """
        logging.debug(f"Raw LLM response: {x}")
        match = re.search(r'\{[^{}]*\}', x, re.DOTALL)
        if match:
            return match.group(0)
        # Fallback: strip common markdown fencing
        x = x.strip().strip("`")
        if x.startswith("json"):
            x = x[4:].strip()
        return x

    def ocr_page_images(self, images: list) -> str:
        """Extract text from PDF page images using the OCR model.

        Used as a fallback when PyPDF cannot extract text from a page
        (e.g. scanned or image-based PDFs). Each image's .data bytes are sent
        to the OCR model and results are joined.

        :param images: List of pypdf ImageFile objects (must have a .data attribute)
        :type images: list
        :return: Extracted text from all images on the page
        :rtype: str
        """
        logging.info(f"Running OCR with model {self.OCR_MODEL} on {len(images)} image(s)...")
        text_parts = []
        for img in images:
            response = self.client.chat(
                model=self.OCR_MODEL,
                messages=[{
                    "role": "user",
                    "content": self.OCR_MODEL_PROMPT,
                    "images": [img.data],
                }],
            )
            text = response["message"]["content"].strip()
            if text:
                text_parts.append(text)
        return "\n".join(text_parts)

    def summarize_text(self, full_text: str) -> dict:
        """Create a 1-2 paragraph abstract from document text."""
        logging.info(f"Summarizing with model {self.SUMMARY_MODEL}...")
        response = self.client.chat(
            model=self.SUMMARY_MODEL,
            messages=[
                {"role": "system", "content": self.SUMMARY_MODEL_PROMPT},
                {"role": "user", "content": full_text},
            ],
        )
        try:
            t = Summary.model_validate_json(
                self.json_loads_with_stringify(response["message"]["content"])
            )
        except ValidationError as e:
            logging.error(f"Failed to synthesize summary from ollama response: {e}")
            logging.error(
                f"Failed to synthesize summary from ollama response: {response['message']['content']}"
            )
            t = Summary(summary="")
        return t.model_dump(mode="json")

    def llm_authors(self, x: list[str]) -> dict:
        """Extract author names from the first lines of a document."""
        logging.info(f"Getting authors with model {self.AUTHORS_MODEL}...")
        full_text = "\n".join(x)
        response = self.client.chat(
            model=self.AUTHORS_MODEL,
            messages=[
                {"role": "system", "content": self.AUTHORS_MODEL_PROMPT},
                {"role": "user", "content": full_text},
            ],
        )
        try:
            t = Authors.model_validate_json(
                self.json_loads_with_stringify(response["message"]["content"])
            )
        except ValidationError as e:
            logging.error(f"Failed to synthesize authors from ollama response: {e}")
            logging.error(
                f"Failed to parse authors from ollama response: {response['message']['content']}"
            )
            t = Authors(authors_list=[], authors="")
        return t.model_dump(mode="json")

    def llm_title(self, x: list[str]) -> dict:
        """Extract the document title from the first lines of a document."""
        logging.info(f"Getting title with model {self.TITLE_MODEL}...")
        full_text = "\n".join(x)
        response = self.client.chat(
            model=self.TITLE_MODEL,
            messages=[
                {"role": "system", "content": self.TITLE_MODEL_PROMPT},
                {"role": "user", "content": full_text},
            ],
        )
        try:
            t = Title.model_validate_json(
                self.json_loads_with_stringify(response["message"]["content"])
            )
        except ValidationError as e:
            logging.error(f"Failed to synthesize title from ollama response: {e}")
            logging.error(
                f"Failed to parse title from ollama response: {response['message']['content']}"
            )
            t = Title(title="")
        return t.model_dump(mode="json")
