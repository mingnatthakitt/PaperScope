from pathlib import Path

import fitz
import pytest

from paperscope.processing.pdf import (
    PageText,
    PDFLimitError,
    PDFValidationError,
    chunk_pages,
    detect_sections,
    extract_pages,
    render_pages,
    sanitize_pdf_text,
    sha256_file,
    validate_pdf,
)


def _fixture_pdf(path: Path) -> None:
    document = fitz.open()
    for index, heading in enumerate(["1 Introduction", "2 Method"]):
        page = document.new_page(width=500, height=700)
        page.insert_text((48, 70), heading, fontsize=18)
        page.insert_text((48, 110), "This paragraph explains the proposed method and its evidence. " * 12, fontsize=10)
        page.draw_rect(fitz.Rect(48, 260, 420, 480), color=(0.7, 0.3, 0.2), width=2)
        page.insert_text((70, 300), f"Figure {index + 1}: a vector visual", fontsize=12)
    document.save(path)
    document.close()


def test_pdf_pipeline_preserves_pages_and_hash(tmp_path: Path) -> None:
    source = tmp_path / "fixture.pdf"
    _fixture_pdf(source)

    assert validate_pdf(source, 10_000_000, 10) == 2
    pages = extract_pages(source)
    assert [page.page_number for page in pages] == [1, 2]
    assert sha256_file(source) == sha256_file(source)

    sections = detect_sections(pages)
    chunks = chunk_pages(pages, sections, target_tokens=40, overlap_tokens=8)
    assert chunks
    assert all(chunk.page_start <= chunk.page_end for chunk in chunks)
    assert any("Method" in (chunk.section_title or "") for chunk in chunks)

    rendered = render_pages(source, tmp_path / "pages", dpi=72)
    assert len(rendered) == 2
    assert rendered[0].path.suffix == ".webp"
    assert rendered[0].path.exists()


def test_pdf_validation_distinguishes_invalid_files_and_limits(tmp_path: Path) -> None:
    invalid = tmp_path / "not-a-pdf.bin"
    invalid.write_bytes(b"not a pdf")
    with pytest.raises(PDFValidationError):
        validate_pdf(invalid, 10_000_000, 10)

    source = tmp_path / "limited.pdf"
    _fixture_pdf(source)
    with pytest.raises(PDFLimitError):
        validate_pdf(source, 10_000_000, 1)


def test_section_detection_and_chunking_keep_same_page_headings_separate() -> None:
    pages = [
        PageText(
            page_number=1,
            text="Abstract\nabstract body\n1 Introduction\nintroduction body",
            blocks=["Abstract", "abstract body", "1 Introduction", "introduction body"],
        ),
        PageText(page_number=2, text="2 Method\nmethod body", blocks=["2 Method", "method body"]),
    ]

    sections = detect_sections(pages)

    assert sections == [("Abstract", 1, 1), ("1 Introduction", 1, 1), ("2 Method", 2, 2)]
    assert all(start <= end for _, start, end in sections)
    chunks = chunk_pages(pages, sections, target_tokens=100, overlap_tokens=0)
    assert [chunk.section_title for chunk in chunks] == ["Abstract", "1 Introduction", "2 Method"]
    assert all("\n\n1 Introduction\n\nintroduction body" not in chunk.content for chunk in chunks if chunk.section_title == "Abstract")


def test_pdf_text_sanitization_replaces_postgres_unsupported_nuls() -> None:
    assert sanitize_pdf_text("left\x00right") == "left right"
    assert "\x00" not in sanitize_pdf_text("formula\x00token")
