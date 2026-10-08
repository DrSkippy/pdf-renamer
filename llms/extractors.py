import base64
import logging
import mimetypes
import os
import re
from typing import Any, TypeVar

from openai import OpenAI
from pydantic import BaseModel, ValidationError

T = TypeVar("T", bound=BaseModel)


class Title(BaseModel):
    title: str


class Authors(BaseModel):
    authors_list: list[str]
    authors: str


class Summary(BaseModel):
    summary: str


class OpenRouterExtractors:
    # Every task uses OpenRouter's auto router, which picks a model per prompt.
    # Kept as separate constants so a task can be pinned to a specific model.
    TITLE_MODEL = "openrouter/auto"
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
    AUTHORS_MODEL = "openrouter/auto"
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
    SUMMARY_MODEL = "openrouter/auto"
    SUMMARY_MODEL_PROMPT = (
        "You are a helpful assistant that extracts information from text. The user will "
        "provide you with a text document, and your task is to create a 1-2 paragraph "
        "abstract. Format the result as json with key 'summary'."
    )
    # OCR fallback: used only when PyPDF cannot extract text (scanned/image-based PDFs)
    OCR_MODEL = "openrouter/auto"
    OCR_MODEL_PROMPT = (
        "Extract all text from this image exactly as it appears. "
        "Return only the raw extracted text with no commentary or formatting."
    )
    BASE_URL = "https://openrouter.ai/api/v1"
    API_KEY_ENV = "OPENROUTER_API_KEY"

    def __init__(self) -> None:
        api_key = os.environ.get(self.API_KEY_ENV)
        if not api_key:
            raise RuntimeError(
                f"{self.API_KEY_ENV} is not set; add it to .envrc and run `direnv allow`"
            )
        self.client = OpenAI(base_url=self.BASE_URL, api_key=api_key)
        logging.info(f"Using OpenRouter client against {self.BASE_URL}")

    def json_loads_with_stringify(self, x: str) -> str:
        """Extract a JSON object string from an LLM response.

        Handles chain-of-thought thinking blocks, markdown code fences, prose
        wrappers, and other common LLM response formats. Returns the raw JSON
        string for Pydantic to parse.
        """
        logging.debug(f"Raw LLM response: {x}")
        # Strip chain-of-thought reasoning blocks emitted by some thinking models
        x = re.sub(r"<think>.*?</think>", "", x, flags=re.DOTALL).strip()
        match = re.search(r"\{[^{}]*\}", x, re.DOTALL)
        if match:
            return match.group(0)
        # Fallback: strip common markdown fencing
        x = x.strip().strip("`")
        if x.startswith("json"):
            x = x[4:].strip()
        return x

    def _chat_json(
        self, model: str, system_prompt: str, user_text: str, schema: type[T], empty: T
    ) -> T:
        """Send a chat request constrained to ``schema`` and validate the reply.

        Requests a JSON-schema response format and asks OpenRouter to route only
        to providers that support it. The reply is still parsed leniently, since
        a routed model may wrap the JSON in prose or fences.

        :param model: OpenRouter model id
        :param system_prompt: Task instructions
        :param user_text: Document text
        :param schema: Pydantic model the reply must match
        :param empty: Value returned when the reply cannot be validated
        :return: Validated schema instance, or ``empty`` on failure
        """
        # Strict JSON-schema mode requires closed objects
        json_schema = schema.model_json_schema() | {"additionalProperties": False}
        response = self.client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_text},
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": schema.__name__.lower(),
                    "strict": True,
                    "schema": json_schema,
                },
            },
            extra_body={"provider": {"require_parameters": True}},
        )
        logging.info(f"OpenRouter routed {schema.__name__} request to {response.model}")
        content = response.choices[0].message.content or ""
        try:
            return schema.model_validate_json(self.json_loads_with_stringify(content))
        except ValidationError as e:
            logging.error(f"Failed to parse {schema.__name__} from LLM response: {e}")
            logging.error(f"Unparsed LLM response: {content}")
            return empty

    def ocr_page_images(self, images: list[Any]) -> str:
        """Extract text from PDF page images using the OCR model.

        Used as a fallback when PyPDF cannot extract text from a page
        (e.g. scanned or image-based PDFs). Each image's .data bytes are sent
        to the OCR model as a base64 data URL and results are joined.

        :param images: List of pypdf ImageFile objects (must have .data and .name)
        :type images: list
        :return: Extracted text from all images on the page
        :rtype: str
        """
        logging.info(
            f"Running OCR with model {self.OCR_MODEL} on {len(images)} image(s)..."
        )
        text_parts = []
        for img in images:
            mime = mimetypes.guess_type(str(getattr(img, "name", "")))[0] or "image/png"
            data_url = (
                f"data:{mime};base64,{base64.b64encode(img.data).decode('ascii')}"
            )
            response = self.client.chat.completions.create(
                model=self.OCR_MODEL,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": self.OCR_MODEL_PROMPT},
                            {"type": "image_url", "image_url": {"url": data_url}},
                        ],
                    }
                ],
            )
            text = (response.choices[0].message.content or "").strip()
            if text:
                text_parts.append(text)
        return "\n".join(text_parts)

    def summarize_text(self, full_text: str) -> dict[str, Any]:
        """Create a 1-2 paragraph abstract from document text."""
        logging.info(f"Summarizing with model {self.SUMMARY_MODEL}...")
        t = self._chat_json(
            self.SUMMARY_MODEL,
            self.SUMMARY_MODEL_PROMPT,
            full_text,
            Summary,
            Summary(summary=""),
        )
        return t.model_dump(mode="json")

    def llm_authors(self, x: list[str]) -> dict[str, Any]:
        """Extract author names from the first lines of a document."""
        logging.info(f"Getting authors with model {self.AUTHORS_MODEL}...")
        t = self._chat_json(
            self.AUTHORS_MODEL,
            self.AUTHORS_MODEL_PROMPT,
            "\n".join(x),
            Authors,
            Authors(authors_list=[], authors=""),
        )
        return t.model_dump(mode="json")

    def llm_title(self, x: list[str]) -> dict[str, Any]:
        """Extract the document title from the first lines of a document."""
        logging.info(f"Getting title with model {self.TITLE_MODEL}...")
        t = self._chat_json(
            self.TITLE_MODEL,
            self.TITLE_MODEL_PROMPT,
            "\n".join(x),
            Title,
            Title(title=""),
        )
        return t.model_dump(mode="json")
