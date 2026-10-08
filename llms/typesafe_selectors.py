import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from dateparser.search import search_dates
from typesafe_sdk import Choice, TypeSafeClient

NONE = "none"


@dataclass(frozen=True)
class DateCandidate:
    """A date string found by dateparser, with the line it came from."""

    text: str
    parsed: datetime
    line: str


def title_candidates(lines: list[str], max_span: int = 3) -> list[str]:
    """Build candidate title spans from the opening lines of a document.

    Each candidate is a single line or a run of up to ``max_span`` adjacent lines
    joined with a space, so titles that wrap across lines are still selectable.
    Order follows the document; duplicates are dropped.

    :param lines: Cleaned text lines from the start of the document
    :param max_span: Maximum number of adjacent lines joined into one candidate
    :return: Candidate spans, copied verbatim from the source lines
    """
    seen: dict[str, None] = {}
    for start in range(len(lines)):
        for width in range(1, max_span + 1):
            if start + width > len(lines):
                break
            span = " ".join(lines[start : start + width])
            if span and span.lower() != NONE:
                seen.setdefault(span, None)
    return list(seen)


def date_candidates(lines: list[str]) -> dict[str, DateCandidate]:
    """Find every dateparser match in the given lines.

    :param lines: Cleaned text lines from the start of the document
    :return: Mapping of a unique option label to its DateCandidate
    """
    found: dict[str, DateCandidate] = {}
    for number, line in enumerate(lines, start=1):
        for text, parsed in search_dates(line) or []:
            label = f'"{text}" (line {number})'
            found.setdefault(label, DateCandidate(text=text, parsed=parsed, line=line))
    return found


class TypeSafeSelectors:
    """Select the title and date from source text with TypeSafe Choice questions.

    Instead of asking a generative model to write the title, code enumerates
    candidate spans and dates, and TypeSafe picks among them. The result is always
    copied from the document, or NONE when nothing fits.
    """

    MODEL = "jev-latest"
    # Starting points to tune against real runs; confidence is recorded in the plan.
    TITLE_MIN_CONFIDENCE = 0.5
    DATE_MIN_CONFIDENCE = 0.5
    MAX_OPTIONS = 254  # API allows 255 per Choice; one is reserved for NONE

    TITLE_INSTRUCTIONS = (
        "`document_start` holds the first lines of a PDF document, one text line per line. "
        "Which option is the document's full main title, exactly as it appears? "
        "Titles often wrap across two or three lines; prefer the option that contains the "
        "whole title and nothing else. Do not choose author names, affiliations, journal or "
        "conference names, running headers, section headings, or sentences from the abstract."
    )
    TITLE_NONE = "None of these is the document's main title."
    DATE_INSTRUCTIONS = (
        "`document_start` holds the first lines of a PDF document. Each option is a piece of "
        "text that a date parser recognized as a date, with the line it appears on. "
        "Which option is the date this document was published, released, or written? "
        "Do not choose received/revised/accessed dates when a publication date is present, "
        "years inside citations, or ordinary words and numbers the parser misread as dates."
    )
    DATE_NONE = "None of these is the document's publication date."

    def __init__(self) -> None:
        logging.info(f"Using TypeSafe model {self.MODEL}")

    def select_title_and_date(
        self, lines: list[str]
    ) -> tuple[dict[str, Any], dict[str, Any] | None]:
        """Pick the title span and publication date in one TypeSafe request.

        :param lines: Cleaned text lines from the start of the document
        :return: Tuple of (title_dict, date_dict or None). ``title_dict["title"]`` is
            empty when TypeSafe chose NONE or was below the confidence threshold, so
            callers fall back to the original filename.
        """
        titles = title_candidates(lines)[: self.MAX_OPTIONS]
        dates = dict(list(date_candidates(lines).items())[: self.MAX_OPTIONS])

        questions: dict[str, Choice] = {}
        if titles:
            questions["title"] = Choice(
                instructions=self.TITLE_INSTRUCTIONS,
                criteria={t: None for t in titles} | {NONE: self.TITLE_NONE},
            )
        if dates:
            questions["date"] = Choice(
                instructions=self.DATE_INSTRUCTIONS,
                criteria={label: f"Line: {c.line}" for label, c in dates.items()}
                | {NONE: self.DATE_NONE},
            )
        if not questions:
            return {"title": "", "source": "typesafe"}, None

        logging.info(
            f"Asking TypeSafe: {len(titles)} title candidates, {len(dates)} date candidates"
        )
        with TypeSafeClient(model=self.MODEL) as client:
            response = client.system_one(
                state={"document_start": "\n".join(lines)}, questions=questions
            )
        logging.debug(f"TypeSafe usage: {response.usage}")

        title: dict[str, Any] = {"title": "", "source": "typesafe"}
        if "title" in response.choices:
            answer = response.choices["title"]
            selected = "" if answer.choice == NONE else answer.choice
            title |= {"selected": selected, "confidence": answer.confidence}
            if selected and answer.confidence >= self.TITLE_MIN_CONFIDENCE:
                title["title"] = selected
            else:
                logging.warning(
                    f"Title not accepted (choice={answer.choice!r}, "
                    f"confidence={answer.confidence:.2f})"
                )

        date = None
        if "date" in response.choices:
            answer = response.choices["date"]
            if answer.choice != NONE and answer.confidence >= self.DATE_MIN_CONFIDENCE:
                c = dates[answer.choice]
                date = {
                    "date": str((c.text, c.parsed)),
                    "date_line": c.line,
                    "confidence": answer.confidence,
                    "source": "typesafe",
                }
            else:
                logging.info(
                    f"Date not accepted (choice={answer.choice!r}, "
                    f"confidence={answer.confidence:.2f})"
                )
        return title, date
