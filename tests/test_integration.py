"""Integration tests for end-to-end title extraction against sample PDFs.

These tests require a live Ollama instance at the configured host and the
sample PDFs in the samples/ directory. Run with:

    poetry run pytest -m integration tests/test_integration.py -v
"""
import pytest
from pathlib import Path

from utils.pdf_content import extract_from_pdf

SAMPLES = Path(__file__).parent.parent / "samples"


@pytest.mark.integration
@pytest.mark.parametrize(
    "filename,required_terms",
    [
        (
            "boots-slides.pdf",
            ["spectral algorithms", "dynamical systems"],
        ),
        (
            "ggordon-slides.pdf",
            ["spectral learning"],
        ),
        (
            "Luxburg07_tutorial_4488[0].pdf",
            ["tutorial", "spectral clustering"],
        ),
        (
            "song-slides.pdf",
            ["spectral algorithms", "latent tree"],
        ),
        (
            "autoscalingcloudinfrwithrl.pdf",
            ["auto-scaling", "reinforcement learning"],
        ),
    ],
)
def test_title_extraction(filename: str, required_terms: list[str]) -> None:
    """Extracted title must contain all required terms (case-insensitive)."""
    pdf_path = SAMPLES / filename
    assert pdf_path.exists(), f"Sample file not found: {pdf_path}"

    title, _, _, _ = extract_from_pdf(pdf_path)
    extracted = title["title"].lower()

    for term in required_terms:
        assert term in extracted, (
            f"Expected '{term}' in extracted title, got: {title['title']!r}"
        )
