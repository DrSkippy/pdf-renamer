"""Unit tests for bin/pdf-renamer.py processing loops."""
import importlib.util
import json
import logging
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

# bin/pdf-renamer.py has a hyphen so it cannot be imported with normal import syntax.
_BIN = Path(__file__).resolve().parent.parent / "bin" / "pdf-renamer.py"
_spec = importlib.util.spec_from_file_location("pdf_renamer", _BIN)
renamer = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(renamer)


GOOD_RESULT = (
    {"title": "Good Title"},
    {"authors": "Jane Doe", "authors_list": ["Jane Doe"]},
    None,
    {"summary": "A good summary."},
)
EMPTY_TITLE_RESULT = (
    {"title": ""},
    {"authors": "", "authors_list": []},
    None,
    {"summary": ""},
)


@pytest.fixture()
def pdf_root(tmp_path):
    """Create a temporary directory with two dummy PDF files."""
    (tmp_path / "good.pdf").touch()
    (tmp_path / "bad.pdf").touch()
    return tmp_path


class TestRunDryRun:
    def test_continues_after_exception(self, pdf_root, tmp_path, capsys):
        """run_dry_run skips a file that raises an exception and processes the rest."""
        plan_file = tmp_path / "plan.json"

        def fake_extract(path):
            if path.name == "bad.pdf":
                raise Exception("failed to create seqence")
            return GOOD_RESULT

        with patch.object(renamer, "extract_from_pdf", side_effect=fake_extract):
            renamer.run_dry_run(pdf_root, plan_file)

        plan = json.loads(plan_file.read_text())
        titles = [e["title"]["title"] for e in plan]
        assert "Good Title" in titles
        assert len(plan) == 1  # only the successful file

        captured = capsys.readouterr()
        assert "ERROR" in captured.out
        assert "bad.pdf" in captured.out

    def test_error_is_logged(self, pdf_root, tmp_path):
        """run_dry_run logs the exception with exc_info when a file fails."""
        plan_file = tmp_path / "plan.json"

        with patch.object(
            renamer, "extract_from_pdf", side_effect=Exception("boom")
        ), patch.object(renamer.logging, "error") as mock_log:
            renamer.run_dry_run(pdf_root, plan_file)

        assert mock_log.called
        logged_msg = mock_log.call_args[0][0]
        assert "boom" in logged_msg or "Failed" in logged_msg

    def test_all_succeed(self, pdf_root, tmp_path):
        """run_dry_run adds all files to the plan when none fail."""
        plan_file = tmp_path / "plan.json"

        with patch.object(renamer, "extract_from_pdf", return_value=GOOD_RESULT):
            count = renamer.run_dry_run(pdf_root, plan_file)

        assert count == 2
        plan = json.loads(plan_file.read_text())
        assert len(plan) == 2


class TestRunFull:
    def test_continues_after_exception(self, pdf_root, capsys):
        """run_full skips a file that raises an exception and renames the rest."""
        def fake_extract(path):
            if path.name == "bad.pdf":
                raise Exception("failed to create seqence")
            return GOOD_RESULT

        with patch.object(renamer, "extract_from_pdf", side_effect=fake_extract):
            renamed, skipped = renamer.run_full(pdf_root)

        assert renamed == 1
        captured = capsys.readouterr()
        assert "ERROR" in captured.out
        assert "bad.pdf" in captured.out

    def test_all_succeed(self, pdf_root, capsys):
        """run_full renames every PDF when none fail."""
        results = [
            ({"title": "Title One"}, {"authors": "", "authors_list": []}, None, {"summary": ""}),
            ({"title": "Title Two"}, {"authors": "", "authors_list": []}, None, {"summary": ""}),
        ]

        with patch.object(renamer, "extract_from_pdf", side_effect=results):
            renamed, skipped = renamer.run_full(pdf_root)

        assert renamed == 2
        assert skipped == 0

    def test_writes_json_when_output_dir_given(self, pdf_root, tmp_path, capsys):
        """run_full writes one JSON metadata file per renamed PDF when output_dir is set."""
        output_dir = tmp_path / "meta"
        results = [
            ({"title": "Title One"}, {"authors": "", "authors_list": []}, None, {"summary": ""}),
            ({"title": "Title Two"}, {"authors": "", "authors_list": []}, None, {"summary": ""}),
        ]

        with patch.object(renamer, "extract_from_pdf", side_effect=results):
            renamed, skipped = renamer.run_full(pdf_root, output_dir)

        assert renamed == 2
        json_files = list(output_dir.glob("*.json"))
        assert len(json_files) == 2

    def test_skips_when_destination_exists(self, pdf_root, capsys):
        """run_full skips a file when the destination filename already exists."""
        # Both PDFs return the same title → second rename would collide
        with patch.object(renamer, "extract_from_pdf", return_value=GOOD_RESULT):
            renamed, skipped = renamer.run_full(pdf_root)

        # One succeeds, one is skipped due to collision
        assert renamed == 1
        assert skipped == 1
        captured = capsys.readouterr()
        assert "SKIP" in captured.out
