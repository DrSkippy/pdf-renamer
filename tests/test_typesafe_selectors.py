from datetime import datetime
from unittest.mock import MagicMock, Mock, patch

import pytest
from typesafe_sdk import TypeSafeError

from llms.typesafe_selectors import (
    NONE,
    TypeSafeSelectors,
    date_candidates,
    title_candidates,
)
from utils.pdf_content import likely_title


def _answer(choice: str, confidence: float) -> Mock:
    return Mock(choice=choice, confidence=confidence)


def _patched_client(choices: dict) -> tuple:
    """Patch TypeSafeClient so system_one returns the given choice answers."""
    client = MagicMock()
    client.__enter__.return_value = client
    client.system_one.return_value = Mock(choices=choices, usage=None)
    return patch("llms.typesafe_selectors.TypeSafeClient", return_value=client), client


class TestTitleCandidates:
    def test_includes_single_lines_and_adjacent_joins(self):
        lines = ["Deep Learning for", "Natural Language Processing", "Jane Smith"]
        result = title_candidates(lines)

        assert "Deep Learning for" in result
        assert "Deep Learning for Natural Language Processing" in result
        assert "Deep Learning for Natural Language Processing Jane Smith" in result
        assert "Natural Language Processing Jane Smith" in result

    def test_respects_max_span(self):
        lines = ["a1", "b2", "c3", "d4"]
        result = title_candidates(lines, max_span=2)

        assert "a1 b2" in result
        assert "a1 b2 c3" not in result

    def test_deduplicates_preserving_order(self):
        result = title_candidates(["Repeated Header", "Repeated Header"], max_span=1)
        assert result == ["Repeated Header"]

    def test_excludes_span_equal_to_none_label(self):
        assert NONE not in title_candidates(["None"], max_span=1)

    def test_empty(self):
        assert title_candidates([]) == []


class TestDateCandidates:
    @patch("llms.typesafe_selectors.search_dates")
    def test_labels_every_match_with_line_number(self, mock_search_dates):
        mock_search_dates.side_effect = [
            [("Received 2019", datetime(2019, 1, 1))],
            None,
            [("June 2020", datetime(2020, 6, 1)), ("2021", datetime(2021, 1, 1))],
        ]
        result = date_candidates(
            ["Received 2019", "no dates here", "Published June 2020, 2021"]
        )

        assert list(result) == [
            '"Received 2019" (line 1)',
            '"June 2020" (line 3)',
            '"2021" (line 3)',
        ]
        assert result['"June 2020" (line 3)'].line == "Published June 2020, 2021"


class TestSelectTitleAndDate:
    LINES = ["A Great Paper Title", "Jane Smith", "Published June 2020"]

    @patch("llms.typesafe_selectors.search_dates")
    def test_accepts_confident_title_and_date(self, mock_search_dates):
        mock_search_dates.side_effect = [
            None,
            None,
            [("June 2020", datetime(2020, 6, 1))],
        ]
        patcher, client = _patched_client(
            {
                "title": _answer("A Great Paper Title", 0.9),
                "date": _answer('"June 2020" (line 3)', 0.8),
            }
        )
        with patcher:
            title, date = TypeSafeSelectors().select_title_and_date(self.LINES)

        assert title["title"] == "A Great Paper Title"
        assert title["confidence"] == 0.9
        assert date["date_line"] == "Published June 2020"
        assert date["date"] == str(("June 2020", datetime(2020, 6, 1)))

        questions = client.system_one.call_args.kwargs["questions"]
        assert NONE in questions["title"].criteria
        assert NONE in questions["date"].criteria

    @patch("llms.typesafe_selectors.search_dates", return_value=None)
    def test_low_confidence_title_left_empty_for_filename_fallback(self, _):
        patcher, _client = _patched_client({"title": _answer("Jane Smith", 0.2)})
        with patcher:
            title, date = TypeSafeSelectors().select_title_and_date(self.LINES)

        assert title["title"] == ""
        assert title["selected"] == "Jane Smith"
        assert date is None

    @patch("llms.typesafe_selectors.search_dates", return_value=None)
    def test_none_choice_left_empty(self, _):
        patcher, client = _patched_client({"title": _answer(NONE, 0.95)})
        with patcher:
            title, _ = TypeSafeSelectors().select_title_and_date(self.LINES)

        assert title["title"] == ""
        assert "date" not in client.system_one.call_args.kwargs["questions"]

    @patch("llms.typesafe_selectors.search_dates")
    def test_none_date_returns_none(self, mock_search_dates):
        mock_search_dates.side_effect = [[("2019", datetime(2019, 1, 1))], None, None]
        patcher, _client = _patched_client(
            {
                "title": _answer("A Great Paper Title", 0.9),
                "date": _answer(NONE, 0.9),
            }
        )
        with patcher:
            _, date = TypeSafeSelectors().select_title_and_date(self.LINES)

        assert date is None

    def test_no_lines_skips_request(self):
        patcher, client = _patched_client({})
        with patcher:
            title, date = TypeSafeSelectors().select_title_and_date([])

        assert title["title"] == ""
        assert date is None
        client.system_one.assert_not_called()


class TestLikelyTitleWithSelector:
    def test_uses_selector_and_llm_authors(self):
        extractor = Mock()
        extractor.llm_authors.return_value = {
            "authors": "Jane Smith",
            "authors_list": ["Jane Smith"],
        }
        selector = Mock()
        selector.select_title_and_date.return_value = ({"title": "T"}, {"date": "d"})

        title, authors, date = likely_title(["T", "Jane Smith"], extractor, selector)

        assert title == {"title": "T"}
        assert date == {"date": "d"}
        assert authors["authors"] == "Jane Smith"
        extractor.llm_title.assert_not_called()

    @patch("utils.pdf_content.search_dates", return_value=None)
    def test_falls_back_to_llm_on_typesafe_error(self, _):
        extractor = Mock()
        extractor.llm_title.return_value = {"title": "LLM Title"}
        extractor.llm_authors.return_value = {"authors": "", "authors_list": []}
        selector = Mock()
        selector.select_title_and_date.side_effect = TypeSafeError("service down")

        title, _, _ = likely_title(["LLM Title"], extractor, selector)

        assert title == {"title": "LLM Title", "source": "llm-fallback"}
