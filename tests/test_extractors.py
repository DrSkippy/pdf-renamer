import json
import pytest
from unittest.mock import Mock, patch, MagicMock
from llms.extractors import OllamaExtractors


class TestOllamaExtractors:
    """Test suite for OllamaExtractors class"""

    @patch('llms.extractors.ollama.Client')
    def test_init_creates_client(self, mock_client_class):
        """Test that __init__ creates an ollama client with correct host"""
        mock_client_instance = Mock()
        mock_client_class.return_value = mock_client_instance

        extractor = OllamaExtractors()

        mock_client_class.assert_called_once_with(host=OllamaExtractors.HOST)
        assert extractor.client == mock_client_instance

    @patch('llms.extractors.ollama.Client')
    def test_json_loads_with_stringify_basic(self, mock_client_class):
        """Test json_loads_with_stringify with basic JSON"""
        mock_client_class.return_value = Mock()
        extractor = OllamaExtractors()
        json_str = '{"title": "Test Document"}'
        result = extractor.json_loads_with_stringify(json_str)
        assert result == {"title": "Test Document"}

    @patch('llms.extractors.ollama.Client')
    def test_json_loads_with_stringify_with_whitespace(self, mock_client_class):
        """Test json_loads_with_stringify strips whitespace"""
        mock_client_class.return_value = Mock()
        extractor = OllamaExtractors()
        json_str = '  {"title": "Test"}  '
        result = extractor.json_loads_with_stringify(json_str)
        assert result == {"title": "Test"}

    @patch('llms.extractors.ollama.Client')
    def test_json_loads_with_stringify_with_backticks(self, mock_client_class):
        """Test json_loads_with_stringify removes backticks"""
        mock_client_class.return_value = Mock()
        extractor = OllamaExtractors()
        json_str = '```{"title": "Test"}```'
        result = extractor.json_loads_with_stringify(json_str)
        assert result == {"title": "Test"}

    @patch('llms.extractors.ollama.Client')
    def test_json_loads_with_stringify_with_json_prefix(self, mock_client_class):
        """Test json_loads_with_stringify removes 'json' prefix"""
        mock_client_class.return_value = Mock()
        extractor = OllamaExtractors()
        json_str = 'json{"title": "Test"}'
        result = extractor.json_loads_with_stringify(json_str)
        assert result == {"title": "Test"}

    @patch('llms.extractors.ollama.Client')
    def test_json_loads_with_stringify_with_escaped_quotes(self, mock_client_class):
        """Test json_loads_with_stringify handles escaped quotes"""
        mock_client_class.return_value = Mock()
        extractor = OllamaExtractors()
        json_str = '{"title": "Test \\"quoted\\" text"}'
        result = extractor.json_loads_with_stringify(json_str)
        assert result == {"title": "Test 'quoted' text"}

    @patch('llms.extractors.ollama.Client')
    def test_json_loads_with_stringify_with_backslashes(self, mock_client_class):
        """Test json_loads_with_stringify handles backslashes"""
        mock_client_class.return_value = Mock()
        extractor = OllamaExtractors()
        json_str = '{"path": "C:\\\\Users\\\\test"}'
        result = extractor.json_loads_with_stringify(json_str)
        # After escaping, backslashes should be doubled
        assert "path" in result

    @patch('llms.extractors.ollama.Client')
    def test_summarize_text(self, mock_client_class):
        """Test summarize_text method"""
        mock_client = Mock()
        mock_client_class.return_value = mock_client

        # Mock the response from ollama
        mock_response = {
            "message": {
                "content": '{"abstract": "This is a test summary of the document."}'
            }
        }
        mock_client.chat.return_value = mock_response

        extractor = OllamaExtractors()
        result = extractor.summarize_text("Full text of the document")

        # Verify the client was called with correct parameters
        mock_client.chat.assert_called_once()
        call_args = mock_client.chat.call_args
        assert call_args[1]['model'] == OllamaExtractors.SUMMARY_MODEL
        assert len(call_args[1]['messages']) == 2
        assert call_args[1]['messages'][0]['role'] == 'system'
        assert call_args[1]['messages'][1]['role'] == 'user'
        assert call_args[1]['messages'][1]['content'] == "Full text of the document"

        # Verify the result
        assert result == {"abstract": "This is a test summary of the document."}

    @patch('llms.extractors.ollama.Client')
    def test_llm_authors(self, mock_client_class):
        """Test llm_authors method"""
        mock_client = Mock()
        mock_client_class.return_value = mock_client

        # Mock the response from ollama
        mock_response = {
            "message": {
                "content": '{"line_number": 3, "line": "John Doe, Jane Smith", "authors": ["John Doe", "Jane Smith"]}'
            }
        }
        mock_client.chat.return_value = mock_response

        extractor = OllamaExtractors()
        text_lines = ["Title Line", "Date: 2024", "John Doe, Jane Smith", "Abstract..."]
        result = extractor.llm_authors(text_lines)

        # Verify the client was called with correct parameters
        mock_client.chat.assert_called_once()
        call_args = mock_client.chat.call_args
        assert call_args[1]['model'] == OllamaExtractors.AUTHORS_MODEL
        assert len(call_args[1]['messages']) == 2
        assert call_args[1]['messages'][0]['role'] == 'system'
        assert call_args[1]['messages'][1]['role'] == 'user'
        # Input is joined with newlines
        assert "\n" in call_args[1]['messages'][1]['content']

        # Verify the result
        assert result['line_number'] == 3
        assert result['authors'] == ["John Doe", "Jane Smith"]

    @patch('llms.extractors.ollama.Client')
    def test_llm_title(self, mock_client_class):
        """Test llm_title method"""
        mock_client = Mock()
        mock_client_class.return_value = mock_client

        # Mock the response from ollama
        mock_response = {
            "message": {
                "content": '{"line_number": 1, "title": "Advanced Machine Learning Techniques"}'
            }
        }
        mock_client.chat.return_value = mock_response

        extractor = OllamaExtractors()
        text_lines = ["Advanced Machine Learning", "Techniques", "John Doe", "2024"]
        result = extractor.llm_title(text_lines)

        # Verify the client was called with correct parameters
        mock_client.chat.assert_called_once()
        call_args = mock_client.chat.call_args
        assert call_args[1]['model'] == OllamaExtractors.TITLE_MODEL
        assert len(call_args[1]['messages']) == 2
        assert call_args[1]['messages'][0]['role'] == 'system'
        assert call_args[1]['messages'][1]['role'] == 'user'
        # Input is joined with spaces
        assert " " in call_args[1]['messages'][1]['content']

        # Verify the result
        assert result['line_number'] == 1
        assert result['title'] == "Advanced Machine Learning Techniques"

    @patch('llms.extractors.ollama.Client')
    def test_summarize_text_with_markdown_response(self, mock_client_class):
        """Test summarize_text handles markdown-wrapped JSON"""
        mock_client = Mock()
        mock_client_class.return_value = mock_client

        # Mock response with markdown code block
        mock_response = {
            "message": {
                "content": '```json\n{"abstract": "Test abstract"}\n```'
            }
        }
        mock_client.chat.return_value = mock_response

        extractor = OllamaExtractors()
        result = extractor.summarize_text("Test text")

        assert result == {"abstract": "Test abstract"}

    @patch('llms.extractors.ollama.Client')
    def test_methods_use_correct_models(self, mock_client_class):
        """Test that each method uses the correct model"""
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        mock_client.chat.return_value = {
            "message": {"content": '{"result": "test"}'}
        }

        extractor = OllamaExtractors()

        # Test summarize_text uses SUMMARY_MODEL
        extractor.summarize_text("text")
        assert mock_client.chat.call_args[1]['model'] == OllamaExtractors.SUMMARY_MODEL

        # Test llm_authors uses AUTHORS_MODEL
        extractor.llm_authors(["text"])
        assert mock_client.chat.call_args[1]['model'] == OllamaExtractors.AUTHORS_MODEL

        # Test llm_title uses TITLE_MODEL
        extractor.llm_title(["text"])
        assert mock_client.chat.call_args[1]['model'] == OllamaExtractors.TITLE_MODEL


class TestOllamaExtractorsEdgeCases:
    """Test edge cases and error scenarios"""

    @patch('llms.extractors.ollama.Client')
    def test_json_loads_with_stringify_invalid_json(self, mock_client_class):
        """Test json_loads_with_stringify with invalid JSON raises error"""
        mock_client_class.return_value = Mock()
        extractor = OllamaExtractors()
        invalid_json = '{invalid json}'
        with pytest.raises(json.JSONDecodeError):
            extractor.json_loads_with_stringify(invalid_json)

    @patch('llms.extractors.ollama.Client')
    def test_empty_text_to_summarize(self, mock_client_class):
        """Test summarize_text with empty string"""
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        mock_client.chat.return_value = {
            "message": {"content": '{"abstract": ""}'}
        }

        extractor = OllamaExtractors()
        result = extractor.summarize_text("")

        assert result == {"abstract": ""}

    @patch('llms.extractors.ollama.Client')
    def test_empty_list_to_llm_authors(self, mock_client_class):
        """Test llm_authors with empty list"""
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        mock_client.chat.return_value = {
            "message": {"content": '{"authors": []}'}
        }

        extractor = OllamaExtractors()
        result = extractor.llm_authors([])

        # Should join empty list to empty string
        call_args = mock_client.chat.call_args
        assert call_args[1]['messages'][1]['content'] == ""

    @patch('llms.extractors.ollama.Client')
    def test_empty_list_to_llm_title(self, mock_client_class):
        """Test llm_title with empty list"""
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        mock_client.chat.return_value = {
            "message": {"content": '{"title": ""}'}
        }

        extractor = OllamaExtractors()
        result = extractor.llm_title([])

        # Should join empty list to empty string
        call_args = mock_client.chat.call_args
        assert call_args[1]['messages'][1]['content'] == ""
