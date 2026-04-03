import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

from utils.pdf_content import (
    clean_text,
    likely_title,
    extract_from_pdf,
    _extract_page_text,
    MIN_LINE_CHAR_THRESHOLD,
    MIN_CONTENT_LINES,
    MAX_PAGES_TO_READ,
    MAX_SUMMARY_CHARS,
    MIN_OCR_TRIGGER_CHARS,
    MAX_LINES_FOR_TITLE_AND_AUTHORS,
)


class TestCleanText:
    """Test suite for clean_text function"""

    def test_clean_text_basic(self):
        """Test clean_text keeps long lines and drops short ones."""
        raw_text = "This is a long line that should be kept\nSh\nAnother long line to be kept"
        result = clean_text(raw_text)

        assert "This is a long line that should be kept" in result
        assert "Another long line to be kept" in result
        assert "Sh" not in result

    def test_clean_text_filters_short_lines(self):
        """Test that lines at or below threshold are filtered out."""
        raw_text = "a\nbb\nThis is a longer line"
        result = clean_text(raw_text)

        assert "This is a longer line" in result
        # exactly MIN_LINE_CHAR_THRESHOLD chars — should be filtered (not strictly greater)
        threshold_line = "x" * MIN_LINE_CHAR_THRESHOLD
        assert threshold_line not in result

    def test_clean_text_strips_whitespace(self):
        """Test that whitespace is stripped from each line."""
        raw_text = "  This line has leading spaces  \n\tThis has tabs\t"
        result = clean_text(raw_text)

        assert all(line == line.strip() for line in result)
        assert "This line has leading spaces" in result

    def test_clean_text_empty_string(self):
        raw_text = ""
        assert clean_text(raw_text) == []

    def test_clean_text_only_short_lines(self):
        assert clean_text("a\nb\nc") == []

    def test_clean_text_preserves_order(self):
        raw_text = "First long line\nSecond long line\nThird long line"
        result = clean_text(raw_text)

        assert result[0] == "First long line"
        assert result[1] == "Second long line"
        assert result[2] == "Third long line"

    def test_clean_text_newlines_only(self):
        assert clean_text("\n\n\n") == []

    def test_clean_text_keeps_short_words_in_titles(self):
        """Lines longer than threshold like 'AI' extended words should be kept."""
        # MIN_LINE_CHAR_THRESHOLD is 2, so 3+ char lines are kept
        raw_text = "AI\nIoT\nDeep Learning"
        result = clean_text(raw_text)
        assert "IoT" in result
        assert "Deep Learning" in result

    def test_clean_text_with_pdf_like_input(self):
        """Test with text that looks like a real PDF first page."""
        raw_text = (
            "\nResearch Paper Title Here\n"
            "John Doe, Jane Smith\n"
            "Department of Computer Science\n"
            "January 15, 2024\n"
            "\nAbstract\n\n"
            "This is the abstract paragraph with lots of content.\n"
        )
        result = clean_text(raw_text)

        assert any("Research Paper Title Here" in line for line in result)
        assert any("John Doe, Jane Smith" in line for line in result)
        assert any("abstract paragraph" in line for line in result)


class TestLikelyTitle:
    """Test suite for likely_title function"""

    def _make_extractor(self, title="Test Title", authors="Author A", authors_list=None):
        """Return a mock OllamaExtractors instance."""
        mock_extractor = Mock()
        mock_extractor.llm_title.return_value = {"title": title}
        mock_extractor.llm_authors.return_value = {
            "authors": authors,
            "authors_list": authors_list or [authors],
        }
        return mock_extractor

    @patch("utils.pdf_content.search_dates")
    def test_likely_title_all_lines_sent_to_llm(self, mock_search_dates):
        """All lines up to MAX_LINES are always sent to LLM regardless of date position."""
        mock_search_dates.return_value = None
        extractor = self._make_extractor()

        text_lines = [f"Line {i}" for i in range(5)]
        likely_title(text_lines, extractor)

        call_args = extractor.llm_title.call_args[0][0]
        assert call_args == text_lines

    @patch("utils.pdf_content.search_dates")
    def test_likely_title_no_date_returns_none_date(self, mock_search_dates):
        """When no date is found, date should be None."""
        mock_search_dates.return_value = None
        extractor = self._make_extractor()

        _, _, date = likely_title(["Title", "Author Name", "Abstract here"], extractor)
        assert date is None

    @patch("utils.pdf_content.search_dates")
    def test_likely_title_date_detected(self, mock_search_dates):
        """When a date line exists, it should be returned in the date dict."""
        mock_search_dates.side_effect = [
            None,  # line 0
            [("January 15, 2024", "2024-01-15")],  # line 1 has date
            None,
        ]
        extractor = self._make_extractor()

        text_lines = ["Title Here", "January 15, 2024", "Author Name"]
        _, _, date = likely_title(text_lines, extractor)

        assert date is not None
        assert date["date_line"] == "January 15, 2024"

    @patch("utils.pdf_content.search_dates")
    def test_likely_title_date_on_first_line_still_sends_all_lines(self, mock_search_dates):
        """Even if the date is on line 0, all lines should still go to the LLM."""
        mock_search_dates.side_effect = [
            [("January 1, 2024", "2024-01-01")],  # date on first line
        ]
        extractor = self._make_extractor()

        text_lines = ["January 1, 2024", "Test Title", "Author Name"]
        likely_title(text_lines, extractor)

        # LLM must still receive all lines, not just post-date lines
        call_args = extractor.llm_title.call_args[0][0]
        assert call_args == text_lines

    @patch("utils.pdf_content.search_dates")
    def test_likely_title_respects_max_lines(self, mock_search_dates):
        """Only first MAX_LINES_FOR_TITLE_AND_AUTHORS lines are sent to LLM."""
        mock_search_dates.return_value = None
        extractor = self._make_extractor()

        text_lines = [f"Line {i}" for i in range(MAX_LINES_FOR_TITLE_AND_AUTHORS + 10)]
        likely_title(text_lines, extractor)

        call_args = extractor.llm_title.call_args[0][0]
        assert len(call_args) == MAX_LINES_FOR_TITLE_AND_AUTHORS

    @patch("utils.pdf_content.search_dates")
    def test_likely_title_empty_list(self, mock_search_dates):
        """likely_title with empty input should still call LLM with empty list."""
        mock_search_dates.return_value = None
        extractor = self._make_extractor()

        likely_title([], extractor)

        extractor.llm_title.assert_called_once_with([])
        extractor.llm_authors.assert_called_once_with([])

    @patch("utils.pdf_content.search_dates")
    def test_likely_title_stops_at_first_date(self, mock_search_dates):
        """Date scanning stops after the first date found."""
        mock_search_dates.side_effect = [
            None,
            [("January 1, 2024", "2024-01-01")],
            # Should not be called for line 2 onward
        ]
        extractor = self._make_extractor()

        text_lines = ["Title", "January 1, 2024", "Author"]
        _, _, date = likely_title(text_lines, extractor)

        assert date["date_line"] == "January 1, 2024"
        # search_dates called only twice (stopped after finding date)
        assert mock_search_dates.call_count == 2


class TestExtractPageText:
    """Test suite for _extract_page_text OCR fallback logic."""

    def test_uses_pypdf_text_when_sufficient(self):
        """When PyPDF returns enough text, OCR is not called."""
        mock_page = Mock()
        mock_page.extract_text.return_value = "A" * (MIN_OCR_TRIGGER_CHARS + 10)
        mock_extractor = Mock()

        result = _extract_page_text(mock_page, mock_extractor)

        assert result == "A" * (MIN_OCR_TRIGGER_CHARS + 10)
        mock_extractor.ocr_page_images.assert_not_called()

    def test_ocr_fallback_when_text_empty_and_images_present(self):
        """When PyPDF returns empty text and images exist, OCR is invoked."""
        mock_page = Mock()
        mock_page.extract_text.return_value = ""
        mock_page.images = [Mock(data=b"img")]
        mock_extractor = Mock()
        mock_extractor.ocr_page_images.return_value = "OCR extracted text"

        result = _extract_page_text(mock_page, mock_extractor)

        mock_extractor.ocr_page_images.assert_called_once_with(mock_page.images)
        assert result == "OCR extracted text"

    def test_ocr_fallback_when_text_below_threshold(self):
        """When PyPDF returns fewer than MIN_OCR_TRIGGER_CHARS, OCR is tried."""
        mock_page = Mock()
        mock_page.extract_text.return_value = "short"
        mock_page.images = [Mock(data=b"img")]
        mock_extractor = Mock()
        mock_extractor.ocr_page_images.return_value = "OCR result"

        result = _extract_page_text(mock_page, mock_extractor)

        mock_extractor.ocr_page_images.assert_called_once()
        assert result == "OCR result"

    def test_no_ocr_when_no_images(self):
        """When PyPDF returns little text but no images exist, OCR is skipped."""
        mock_page = Mock()
        mock_page.extract_text.return_value = "tiny"
        mock_page.images = []
        mock_extractor = Mock()

        result = _extract_page_text(mock_page, mock_extractor)

        mock_extractor.ocr_page_images.assert_not_called()
        assert result == "tiny"

    def test_returns_empty_string_when_extract_text_returns_none(self):
        """extract_text() returning None is handled gracefully."""
        mock_page = Mock()
        mock_page.extract_text.return_value = None
        mock_page.images = []
        mock_extractor = Mock()

        result = _extract_page_text(mock_page, mock_extractor)

        assert result == ""


class TestExtractFromPdf:
    """Test suite for extract_from_pdf function"""

    @patch("utils.pdf_content.OllamaExtractors")
    @patch("utils.pdf_content.PdfReader")
    def test_extract_from_pdf_single_page(self, mock_pdf_reader_class, mock_extractor_class):
        """Test successful extraction from a single-page PDF."""
        mock_extractor = Mock()
        mock_extractor.llm_title.return_value = {"title": "Test Title"}
        mock_extractor.llm_authors.return_value = {
            "authors": "John Doe",
            "authors_list": ["John Doe"],
        }
        mock_extractor.summarize_text.return_value = {"summary": "A summary."}
        mock_extractor_class.return_value = mock_extractor

        mock_reader = Mock()
        mock_page = Mock()
        # Provide enough text so the content threshold is met
        mock_page.extract_text.return_value = "Long enough line of text here\n" * 100
        mock_reader.pages = [mock_page]
        mock_pdf_reader_class.return_value = mock_reader

        title, authors, date, summary = extract_from_pdf(Path("/fake/test.pdf"))

        assert title["title"] == "Test Title"
        assert authors["authors"] == "John Doe"
        assert summary["summary"] == "A summary."

    @patch("utils.pdf_content.OllamaExtractors")
    @patch("utils.pdf_content.PdfReader")
    def test_extract_from_pdf_ocr_fallback(self, mock_pdf_reader_class, mock_extractor_class):
        """When first page is image-based, OCR is used and results flow through normally."""
        mock_extractor = Mock()
        mock_extractor.ocr_page_images.return_value = "OCR extracted text\n" * 100
        mock_extractor.llm_title.return_value = {"title": "OCR Title"}
        mock_extractor.llm_authors.return_value = {"authors": "Author", "authors_list": ["Author"]}
        mock_extractor.summarize_text.return_value = {"summary": "A summary."}
        mock_extractor_class.return_value = mock_extractor

        mock_reader = Mock()
        mock_page = Mock()
        mock_page.extract_text.return_value = ""  # no text — scanned page
        mock_page.images = [Mock(data=b"fake image bytes")]
        mock_reader.pages = [mock_page]
        mock_pdf_reader_class.return_value = mock_reader

        title, _, _, _ = extract_from_pdf(Path("/fake/scanned.pdf"))

        mock_extractor.ocr_page_images.assert_called_once()
        assert title["title"] == "OCR Title"

    @patch("utils.pdf_content.OllamaExtractors")
    @patch("utils.pdf_content.PdfReader")
    def test_extract_from_pdf_reads_extra_pages_when_short(
        self, mock_pdf_reader_class, mock_extractor_class
    ):
        """Test that additional pages are read when first page is too short."""
        mock_extractor = Mock()
        mock_extractor.llm_title.return_value = {"title": "Title"}
        mock_extractor.llm_authors.return_value = {"authors": "", "authors_list": []}
        mock_extractor.summarize_text.return_value = {"summary": ""}
        mock_extractor_class.return_value = mock_extractor

        mock_reader = Mock()
        short_page = Mock()
        short_page.extract_text.return_value = "Short line here\n" * 5  # below threshold

        long_page = Mock()
        long_page.extract_text.return_value = "Long enough line of text here\n" * 100

        mock_reader.pages = [short_page, long_page]
        mock_pdf_reader_class.return_value = mock_reader

        extract_from_pdf(Path("/fake/test.pdf"))

        # Both pages should have had extract_text called
        assert short_page.extract_text.call_count == 1
        assert long_page.extract_text.call_count == 1

    @patch("utils.pdf_content.OllamaExtractors")
    @patch("utils.pdf_content.PdfReader")
    def test_extract_from_pdf_respects_max_pages(self, mock_pdf_reader_class, mock_extractor_class):
        """Page reading stops at MAX_PAGES_TO_READ even if MIN_CONTENT_LINES not reached."""
        mock_extractor = Mock()
        mock_extractor.llm_title.return_value = {"title": "T"}
        mock_extractor.llm_authors.return_value = {"authors": "", "authors_list": []}
        mock_extractor.summarize_text.return_value = {"summary": ""}
        mock_extractor_class.return_value = mock_extractor

        mock_reader = Mock()
        short_page = Mock()
        short_page.extract_text.return_value = "Short line\n" * 5
        short_page.images = []
        mock_reader.pages = [short_page] * 10  # 10 pages available
        mock_pdf_reader_class.return_value = mock_reader

        extract_from_pdf(Path("/fake/long.pdf"))

        assert short_page.extract_text.call_count <= MAX_PAGES_TO_READ

    @patch("utils.pdf_content.OllamaExtractors")
    @patch("utils.pdf_content.PdfReader")
    def test_extract_from_pdf_summary_truncated(self, mock_pdf_reader_class, mock_extractor_class):
        """Text sent to summarize_text is capped at MAX_SUMMARY_CHARS."""
        mock_extractor = Mock()
        mock_extractor.llm_title.return_value = {"title": "T"}
        mock_extractor.llm_authors.return_value = {"authors": "", "authors_list": []}
        mock_extractor.summarize_text.return_value = {"summary": ""}
        mock_extractor_class.return_value = mock_extractor

        mock_reader = Mock()
        mock_page = Mock()
        mock_page.extract_text.return_value = "A" * (MAX_SUMMARY_CHARS * 3)
        mock_reader.pages = [mock_page]
        mock_pdf_reader_class.return_value = mock_reader

        extract_from_pdf(Path("/fake/huge.pdf"))

        sent_text = mock_extractor.summarize_text.call_args[0][0]
        assert len(sent_text) <= MAX_SUMMARY_CHARS

    @patch("utils.pdf_content.PdfReader")
    def test_extract_from_pdf_file_not_found(self, mock_pdf_reader_class):
        """Test that FileNotFoundError propagates."""
        mock_pdf_reader_class.side_effect = FileNotFoundError("File not found")

        with pytest.raises(FileNotFoundError):
            extract_from_pdf(Path("/fake/nonexistent.pdf"))

    @patch("utils.pdf_content.PdfReader")
    def test_extract_from_pdf_corrupted_file(self, mock_pdf_reader_class):
        """Test that a corrupt PDF exception propagates."""
        mock_pdf_reader_class.side_effect = Exception("Corrupted PDF")

        with pytest.raises(Exception):
            extract_from_pdf(Path("/fake/corrupted.pdf"))


class TestConstants:
    """Verify module constants have expected values."""

    def test_min_line_char_threshold(self):
        assert MIN_LINE_CHAR_THRESHOLD == 2

    def test_min_content_lines(self):
        assert MIN_CONTENT_LINES == 66 * 8

    def test_max_pages_to_read(self):
        assert MAX_PAGES_TO_READ == 3

    def test_max_summary_chars(self):
        assert MAX_SUMMARY_CHARS == 4000

    def test_min_ocr_trigger_chars(self):
        assert MIN_OCR_TRIGGER_CHARS == 50

    def test_max_lines_for_title_and_authors(self):
        assert MAX_LINES_FOR_TITLE_AND_AUTHORS == 30
