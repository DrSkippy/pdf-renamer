import base64

import pytest
from unittest.mock import Mock, patch
from llms.extractors import OpenRouterExtractors, Title, Authors, Summary


def _reply(content):
    """Build a fake chat completion whose first choice has the given content."""
    return Mock(
        choices=[Mock(message=Mock(content=content))], model="some/routed-model"
    )


@pytest.fixture
def client(monkeypatch):
    """Patch the OpenAI client class and return the mock client instance."""
    monkeypatch.setenv(OpenRouterExtractors.API_KEY_ENV, "test-key")
    with patch("llms.extractors.OpenAI") as mock_client_class:
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        mock_client.class_mock = mock_client_class
        yield mock_client


def _image(data, name):
    """Fake pypdf ImageFile (Mock's own ``name`` kwarg can't set the attribute)."""
    img = Mock(data=data)
    img.name = name
    return img


def _call_kwargs(client):
    return client.chat.completions.create.call_args[1]


class TestOpenRouterExtractors:
    """Test suite for OpenRouterExtractors class"""

    def test_init_creates_client(self, client):
        """Test that __init__ creates an OpenRouter client with the env API key."""
        extractor = OpenRouterExtractors()

        client.class_mock.assert_called_once_with(
            base_url=OpenRouterExtractors.BASE_URL, api_key="test-key"
        )
        assert extractor.client == client

    def test_init_without_api_key_raises(self, monkeypatch):
        """Test that a missing API key fails fast with a pointer to .envrc."""
        monkeypatch.delenv(OpenRouterExtractors.API_KEY_ENV, raising=False)
        with pytest.raises(RuntimeError, match=".envrc"):
            OpenRouterExtractors()

    def test_json_loads_with_stringify_basic(self, client):
        """Test that a plain JSON string is returned unchanged."""
        extractor = OpenRouterExtractors()
        result = extractor.json_loads_with_stringify('{"title": "Test Document"}')
        assert result == '{"title": "Test Document"}'

    def test_json_loads_with_stringify_strips_whitespace(self, client):
        """Test that surrounding whitespace is ignored."""
        extractor = OpenRouterExtractors()
        result = extractor.json_loads_with_stringify('  {"title": "Test"}  ')
        assert result == '{"title": "Test"}'

    def test_json_loads_with_stringify_markdown_fenced(self, client):
        """Test that JSON inside a markdown code fence is extracted."""
        extractor = OpenRouterExtractors()
        result = extractor.json_loads_with_stringify('```json\n{"title": "Test"}\n```')
        assert result == '{"title": "Test"}'

    def test_json_loads_with_stringify_json_prefix(self, client):
        """Test that JSON embedded after a json prefix is extracted."""
        extractor = OpenRouterExtractors()
        result = extractor.json_loads_with_stringify('json{"title": "Test"}')
        assert result == '{"title": "Test"}'

    def test_json_loads_with_stringify_json_in_prose(self, client):
        """Test that JSON embedded in surrounding prose is extracted."""
        extractor = OpenRouterExtractors()
        result = extractor.json_loads_with_stringify(
            'Here is the result: {"title": "Extracted"} as requested.'
        )
        assert result == '{"title": "Extracted"}'

    def test_json_loads_with_stringify_strips_think_block(self, client):
        """Test that <think>...</think> reasoning blocks are stripped before JSON extraction."""
        extractor = OpenRouterExtractors()
        response = '<think>\nThe title is clearly "Spectral Learning".\n</think>\n{"title": "Spectral Learning"}'
        result = extractor.json_loads_with_stringify(response)
        assert result == '{"title": "Spectral Learning"}'

    def test_json_loads_with_stringify_think_block_only(self, client):
        """Test that a response with only a <think> block and no JSON returns empty string."""
        extractor = OpenRouterExtractors()
        result = extractor.json_loads_with_stringify(
            "<think>\nSpectral Learning Part I\n</think>"
        )
        assert result == ""

    def test_json_loads_with_stringify_think_block_with_json_inside(self, client):
        """Test that JSON inside a <think> block is ignored; only post-think JSON is returned."""
        extractor = OpenRouterExtractors()
        response = '<think>Maybe {"title": "wrong"}</think>{"title": "correct"}'
        result = extractor.json_loads_with_stringify(response)
        assert result == '{"title": "correct"}'

    def test_json_loads_with_stringify_preserves_escaped_quotes(self, client):
        """Test that valid JSON escaped quotes are preserved (not mangled)."""
        extractor = OpenRouterExtractors()
        result = extractor.json_loads_with_stringify(
            '{"title": "Test \\"quoted\\" text"}'
        )
        t = Title.model_validate_json(result)
        assert "quoted" in t.title

    def test_summarize_text(self, client):
        """Test summarize_text returns a dict with 'summary' key."""
        client.chat.completions.create.return_value = _reply(
            '{"summary": "This is a test summary."}'
        )

        extractor = OpenRouterExtractors()
        result = extractor.summarize_text("Full text of the document")

        client.chat.completions.create.assert_called_once()
        call_args = _call_kwargs(client)
        assert call_args["model"] == OpenRouterExtractors.SUMMARY_MODEL
        assert call_args["messages"][0]["role"] == "system"
        assert call_args["messages"][1]["content"] == "Full text of the document"
        assert result == {"summary": "This is a test summary."}

    def test_structured_request_uses_strict_json_schema(self, client):
        """Test requests ask for the Pydantic schema and providers that support it."""
        client.chat.completions.create.return_value = _reply('{"summary": "s"}')

        OpenRouterExtractors().summarize_text("text")

        call_args = _call_kwargs(client)
        json_schema = call_args["response_format"]["json_schema"]
        assert call_args["response_format"]["type"] == "json_schema"
        assert json_schema["strict"] is True
        assert (
            json_schema["schema"]["properties"]
            == Summary.model_json_schema()["properties"]
        )
        assert json_schema["schema"]["additionalProperties"] is False
        assert call_args["extra_body"] == {"provider": {"require_parameters": True}}

    def test_summarize_text_validation_error_returns_empty(self, client):
        """Test summarize_text returns empty summary on malformed LLM response."""
        client.chat.completions.create.return_value = _reply("not json at all")
        assert OpenRouterExtractors().summarize_text("text") == {"summary": ""}

    def test_none_content_returns_empty(self, client):
        """Test a reply with no content is treated as a validation failure."""
        client.chat.completions.create.return_value = _reply(None)
        assert OpenRouterExtractors().llm_title(["x"]) == {"title": ""}

    def test_llm_authors(self, client):
        """Test llm_authors returns authors dict without line_number."""
        client.chat.completions.create.return_value = _reply(
            '{"authors": "John Doe, Jane Smith", "authors_list": ["John Doe", "Jane Smith"]}'
        )

        extractor = OpenRouterExtractors()
        text_lines = ["Title Line", "Date: 2024", "John Doe, Jane Smith", "Abstract..."]
        result = extractor.llm_authors(text_lines)

        call_args = _call_kwargs(client)
        assert call_args["model"] == OpenRouterExtractors.AUTHORS_MODEL
        assert call_args["response_format"]["json_schema"]["name"] == "authors"
        # Input joined with newlines (preserves line structure)
        assert "\n" in call_args["messages"][1]["content"]

        assert result["authors"] == "John Doe, Jane Smith"
        assert result["authors_list"] == ["John Doe", "Jane Smith"]
        assert "line_number" not in result

    def test_llm_authors_validation_error_returns_empty(self, client):
        """Test llm_authors returns empty authors on malformed LLM response."""
        client.chat.completions.create.return_value = _reply("bad response")
        result = OpenRouterExtractors().llm_authors(["some lines"])
        assert result == {"authors": "", "authors_list": []}

    def test_llm_title(self, client):
        """Test llm_title returns title dict without line_number."""
        client.chat.completions.create.return_value = _reply(
            '{"title": "Advanced Machine Learning Techniques"}'
        )

        extractor = OpenRouterExtractors()
        text_lines = ["Advanced Machine Learning", "Techniques", "John Doe", "2024"]
        result = extractor.llm_title(text_lines)

        call_args = _call_kwargs(client)
        assert call_args["model"] == OpenRouterExtractors.TITLE_MODEL
        assert call_args["response_format"]["json_schema"]["name"] == "title"
        assert "\n" in call_args["messages"][1]["content"]

        assert result["title"] == "Advanced Machine Learning Techniques"
        assert "line_number" not in result

    def test_llm_title_validation_error_returns_empty(self, client):
        """Test llm_title returns empty title on malformed LLM response."""
        client.chat.completions.create.return_value = _reply("bad response")
        assert OpenRouterExtractors().llm_title(["some lines"]) == {"title": ""}

    def test_all_tasks_use_auto_router(self, client):
        """Every task routes through OpenRouter's auto router."""
        assert {
            OpenRouterExtractors.TITLE_MODEL,
            OpenRouterExtractors.AUTHORS_MODEL,
            OpenRouterExtractors.SUMMARY_MODEL,
            OpenRouterExtractors.OCR_MODEL,
        } == {"openrouter/auto"}

    def test_ocr_page_images_single_image(self, client):
        """Test ocr_page_images sends the image as a base64 data URL."""
        client.chat.completions.create.return_value = _reply(
            "Extracted text from image"
        )

        extractor = OpenRouterExtractors()
        result = extractor.ocr_page_images([_image(b"fake image bytes", "Im0.jpg")])

        call_args = _call_kwargs(client)
        assert call_args["model"] == OpenRouterExtractors.OCR_MODEL
        parts = call_args["messages"][0]["content"]
        assert parts[0] == {
            "type": "text",
            "text": OpenRouterExtractors.OCR_MODEL_PROMPT,
        }
        expected = (
            "data:image/jpeg;base64," + base64.b64encode(b"fake image bytes").decode()
        )
        assert parts[1] == {"type": "image_url", "image_url": {"url": expected}}
        assert result == "Extracted text from image"

    def test_ocr_page_images_unknown_type_defaults_to_png(self, client):
        """Test images without a recognizable extension are sent as PNG."""
        client.chat.completions.create.return_value = _reply("text")

        OpenRouterExtractors().ocr_page_images([_image(b"x", "Im0")])

        url = _call_kwargs(client)["messages"][0]["content"][1]["image_url"]["url"]
        assert url.startswith("data:image/png;base64,")

    def test_ocr_page_images_multiple_images(self, client):
        """Test ocr_page_images concatenates text from multiple images."""
        client.chat.completions.create.side_effect = [
            _reply("First image text"),
            _reply("Second image text"),
        ]

        result = OpenRouterExtractors().ocr_page_images(
            [_image(b"img1", "a.png"), _image(b"img2", "b.png")]
        )

        assert client.chat.completions.create.call_count == 2
        assert "First image text" in result
        assert "Second image text" in result

    def test_ocr_page_images_empty_list(self, client):
        """Test ocr_page_images with no images returns empty string."""
        assert OpenRouterExtractors().ocr_page_images([]) == ""

    def test_empty_list_to_llm_title(self, client):
        """Test llm_title with empty list sends empty string to LLM."""
        client.chat.completions.create.return_value = _reply('{"title": ""}')

        OpenRouterExtractors().llm_title([])

        assert _call_kwargs(client)["messages"][1]["content"] == ""

    def test_empty_list_to_llm_authors(self, client):
        """Test llm_authors with empty list sends empty string to LLM."""
        client.chat.completions.create.return_value = _reply(
            '{"authors": "", "authors_list": []}'
        )

        OpenRouterExtractors().llm_authors([])

        assert _call_kwargs(client)["messages"][1]["content"] == ""
