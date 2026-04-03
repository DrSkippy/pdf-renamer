import pytest
from unittest.mock import Mock, patch
from pydantic import ValidationError
from llms.extractors import OllamaExtractors, Title, Authors, Summary


class TestOllamaExtractors:
    """Test suite for OllamaExtractors class"""

    @patch("llms.extractors.ollama.Client")
    def test_init_creates_client(self, mock_client_class):
        """Test that __init__ creates an ollama client with the configured host."""
        mock_client_instance = Mock()
        mock_client_class.return_value = mock_client_instance

        extractor = OllamaExtractors()

        mock_client_class.assert_called_once_with(host=OllamaExtractors.HOST)
        assert extractor.client == mock_client_instance

    @patch("llms.extractors.ollama.Client")
    def test_json_loads_with_stringify_basic(self, mock_client_class):
        """Test that a plain JSON string is returned unchanged."""
        mock_client_class.return_value = Mock()
        extractor = OllamaExtractors()
        json_str = '{"title": "Test Document"}'
        result = extractor.json_loads_with_stringify(json_str)
        assert result == '{"title": "Test Document"}'

    @patch("llms.extractors.ollama.Client")
    def test_json_loads_with_stringify_strips_whitespace(self, mock_client_class):
        """Test that surrounding whitespace is ignored."""
        mock_client_class.return_value = Mock()
        extractor = OllamaExtractors()
        result = extractor.json_loads_with_stringify('  {"title": "Test"}  ')
        assert result == '{"title": "Test"}'

    @patch("llms.extractors.ollama.Client")
    def test_json_loads_with_stringify_markdown_fenced(self, mock_client_class):
        """Test that JSON inside a markdown code fence is extracted."""
        mock_client_class.return_value = Mock()
        extractor = OllamaExtractors()
        result = extractor.json_loads_with_stringify('```json\n{"title": "Test"}\n```')
        assert result == '{"title": "Test"}'

    @patch("llms.extractors.ollama.Client")
    def test_json_loads_with_stringify_json_prefix(self, mock_client_class):
        """Test that JSON embedded after a json prefix is extracted."""
        mock_client_class.return_value = Mock()
        extractor = OllamaExtractors()
        result = extractor.json_loads_with_stringify('json{"title": "Test"}')
        assert result == '{"title": "Test"}'

    @patch("llms.extractors.ollama.Client")
    def test_json_loads_with_stringify_json_in_prose(self, mock_client_class):
        """Test that JSON embedded in surrounding prose is extracted."""
        mock_client_class.return_value = Mock()
        extractor = OllamaExtractors()
        result = extractor.json_loads_with_stringify(
            'Here is the result: {"title": "Extracted"} as requested.'
        )
        assert result == '{"title": "Extracted"}'

    @patch("llms.extractors.ollama.Client")
    def test_json_loads_with_stringify_strips_think_block(self, mock_client_class):
        """Test that <think>...</think> reasoning blocks are stripped before JSON extraction."""
        mock_client_class.return_value = Mock()
        extractor = OllamaExtractors()
        response = '<think>\nThe title is clearly "Spectral Learning".\n</think>\n{"title": "Spectral Learning"}'
        result = extractor.json_loads_with_stringify(response)
        assert result == '{"title": "Spectral Learning"}'

    @patch("llms.extractors.ollama.Client")
    def test_json_loads_with_stringify_think_block_only(self, mock_client_class):
        """Test that a response with only a <think> block and no JSON returns empty string."""
        mock_client_class.return_value = Mock()
        extractor = OllamaExtractors()
        response = "<think>\nSpectral Learning Part I\n</think>"
        result = extractor.json_loads_with_stringify(response)
        assert result == ""

    @patch("llms.extractors.ollama.Client")
    def test_json_loads_with_stringify_think_block_with_json_inside(self, mock_client_class):
        """Test that JSON inside a <think> block is ignored; only post-think JSON is returned."""
        mock_client_class.return_value = Mock()
        extractor = OllamaExtractors()
        # JSON inside think block should be discarded; real JSON follows
        response = '<think>Maybe {"title": "wrong"}</think>{"title": "correct"}'
        result = extractor.json_loads_with_stringify(response)
        assert result == '{"title": "correct"}'

    @patch("llms.extractors.ollama.Client")
    def test_json_loads_with_stringify_preserves_escaped_quotes(self, mock_client_class):
        """Test that valid JSON escaped quotes are preserved (not mangled)."""
        mock_client_class.return_value = Mock()
        extractor = OllamaExtractors()
        json_str = '{"title": "Test \\"quoted\\" text"}'
        result = extractor.json_loads_with_stringify(json_str)
        # The JSON should be parseable by Pydantic as-is
        t = Title.model_validate_json(result)
        assert "quoted" in t.title

    @patch("llms.extractors.ollama.Client")
    def test_summarize_text(self, mock_client_class):
        """Test summarize_text returns a dict with 'summary' key."""
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        mock_client.chat.return_value = {
            "message": {"content": '{"summary": "This is a test summary."}'}
        }

        extractor = OllamaExtractors()
        result = extractor.summarize_text("Full text of the document")

        mock_client.chat.assert_called_once()
        call_args = mock_client.chat.call_args[1]
        assert call_args["model"] == OllamaExtractors.SUMMARY_MODEL
        assert call_args["format"] == Summary.model_json_schema()
        assert call_args["think"] is False
        assert call_args["messages"][0]["role"] == "system"
        assert call_args["messages"][1]["content"] == "Full text of the document"
        assert result == {"summary": "This is a test summary."}

    @patch("llms.extractors.ollama.Client")
    def test_summarize_text_validation_error_returns_empty(self, mock_client_class):
        """Test summarize_text returns empty summary on malformed LLM response."""
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        mock_client.chat.return_value = {
            "message": {"content": "not json at all"}
        }

        extractor = OllamaExtractors()
        result = extractor.summarize_text("text")
        assert result == {"summary": ""}

    @patch("llms.extractors.ollama.Client")
    def test_llm_authors(self, mock_client_class):
        """Test llm_authors returns authors dict without line_number."""
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        mock_client.chat.return_value = {
            "message": {
                "content": '{"authors": "John Doe, Jane Smith", "authors_list": ["John Doe", "Jane Smith"]}'
            }
        }

        extractor = OllamaExtractors()
        text_lines = ["Title Line", "Date: 2024", "John Doe, Jane Smith", "Abstract..."]
        result = extractor.llm_authors(text_lines)

        call_args = mock_client.chat.call_args[1]
        assert call_args["model"] == OllamaExtractors.AUTHORS_MODEL
        assert call_args["format"] == Authors.model_json_schema()
        assert call_args["think"] is False
        # Input joined with newlines (preserves line structure)
        assert "\n" in call_args["messages"][1]["content"]

        assert result["authors"] == "John Doe, Jane Smith"
        assert result["authors_list"] == ["John Doe", "Jane Smith"]
        assert "line_number" not in result

    @patch("llms.extractors.ollama.Client")
    def test_llm_authors_validation_error_returns_empty(self, mock_client_class):
        """Test llm_authors returns empty authors on malformed LLM response."""
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        mock_client.chat.return_value = {"message": {"content": "bad response"}}

        extractor = OllamaExtractors()
        result = extractor.llm_authors(["some lines"])
        assert result == {"authors": "", "authors_list": []}

    @patch("llms.extractors.ollama.Client")
    def test_llm_title(self, mock_client_class):
        """Test llm_title returns title dict without line_number."""
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        mock_client.chat.return_value = {
            "message": {
                "content": '{"title": "Advanced Machine Learning Techniques"}'
            }
        }

        extractor = OllamaExtractors()
        text_lines = ["Advanced Machine Learning", "Techniques", "John Doe", "2024"]
        result = extractor.llm_title(text_lines)

        call_args = mock_client.chat.call_args[1]
        assert call_args["model"] == OllamaExtractors.TITLE_MODEL
        assert call_args["format"] == Title.model_json_schema()
        assert call_args["think"] is False
        # Input joined with newlines (preserves line structure)
        assert "\n" in call_args["messages"][1]["content"]

        assert result["title"] == "Advanced Machine Learning Techniques"
        assert "line_number" not in result

    @patch("llms.extractors.ollama.Client")
    def test_llm_title_validation_error_returns_empty(self, mock_client_class):
        """Test llm_title returns empty title on malformed LLM response."""
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        mock_client.chat.return_value = {"message": {"content": "bad response"}}

        extractor = OllamaExtractors()
        result = extractor.llm_title(["some lines"])
        assert result == {"title": ""}

    @patch("llms.extractors.ollama.Client")
    def test_methods_use_correct_models(self, mock_client_class):
        """Test that title/authors use TITLE/AUTHORS_MODEL, summary uses SUMMARY_MODEL."""
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        mock_client.chat.return_value = {
            "message": {"content": '{"summary": "s", "title": "t", "authors": "", "authors_list": []}'}
        }

        extractor = OllamaExtractors()

        extractor.summarize_text("text")
        assert mock_client.chat.call_args[1]["model"] == OllamaExtractors.SUMMARY_MODEL

        extractor.llm_authors(["text"])
        assert mock_client.chat.call_args[1]["model"] == OllamaExtractors.AUTHORS_MODEL

        extractor.llm_title(["text"])
        assert mock_client.chat.call_args[1]["model"] == OllamaExtractors.TITLE_MODEL

    @patch("llms.extractors.ollama.Client")
    def test_title_and_authors_use_different_model_than_summary(self, mock_client_class):
        """TITLE_MODEL and AUTHORS_MODEL should differ from SUMMARY_MODEL."""
        mock_client_class.return_value = Mock()
        assert OllamaExtractors.TITLE_MODEL != OllamaExtractors.SUMMARY_MODEL
        assert OllamaExtractors.AUTHORS_MODEL != OllamaExtractors.SUMMARY_MODEL

    @patch("llms.extractors.ollama.Client")
    def test_ocr_model_differs_from_text_models(self, mock_client_class):
        """OCR_MODEL should be distinct from all text-analysis models."""
        mock_client_class.return_value = Mock()
        assert OllamaExtractors.OCR_MODEL != OllamaExtractors.TITLE_MODEL
        assert OllamaExtractors.OCR_MODEL != OllamaExtractors.AUTHORS_MODEL
        assert OllamaExtractors.OCR_MODEL != OllamaExtractors.SUMMARY_MODEL

    @patch("llms.extractors.ollama.Client")
    def test_ocr_page_images_single_image(self, mock_client_class):
        """Test ocr_page_images calls OCR model with image bytes."""
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        mock_client.chat.return_value = {"message": {"content": "Extracted text from image"}}

        extractor = OllamaExtractors()
        mock_img = Mock()
        mock_img.data = b"fake image bytes"
        result = extractor.ocr_page_images([mock_img])

        call_args = mock_client.chat.call_args[1]
        assert call_args["model"] == OllamaExtractors.OCR_MODEL
        assert b"fake image bytes" in call_args["messages"][0]["images"]
        assert result == "Extracted text from image"

    @patch("llms.extractors.ollama.Client")
    def test_ocr_page_images_multiple_images(self, mock_client_class):
        """Test ocr_page_images concatenates text from multiple images."""
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        mock_client.chat.side_effect = [
            {"message": {"content": "First image text"}},
            {"message": {"content": "Second image text"}},
        ]

        extractor = OllamaExtractors()
        result = extractor.ocr_page_images([Mock(data=b"img1"), Mock(data=b"img2")])

        assert mock_client.chat.call_count == 2
        assert "First image text" in result
        assert "Second image text" in result

    @patch("llms.extractors.ollama.Client")
    def test_ocr_page_images_empty_list(self, mock_client_class):
        """Test ocr_page_images with no images returns empty string."""
        mock_client_class.return_value = Mock()
        extractor = OllamaExtractors()
        result = extractor.ocr_page_images([])
        assert result == ""

    @patch("llms.extractors.ollama.Client")
    def test_empty_list_to_llm_title(self, mock_client_class):
        """Test llm_title with empty list sends empty string to LLM."""
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        mock_client.chat.return_value = {"message": {"content": '{"title": ""}'}}

        extractor = OllamaExtractors()
        extractor.llm_title([])

        call_args = mock_client.chat.call_args[1]
        assert call_args["messages"][1]["content"] == ""

    @patch("llms.extractors.ollama.Client")
    def test_empty_list_to_llm_authors(self, mock_client_class):
        """Test llm_authors with empty list sends empty string to LLM."""
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        mock_client.chat.return_value = {
            "message": {"content": '{"authors": "", "authors_list": []}'}
        }

        extractor = OllamaExtractors()
        extractor.llm_authors([])

        call_args = mock_client.chat.call_args[1]
        assert call_args["messages"][1]["content"] == ""
