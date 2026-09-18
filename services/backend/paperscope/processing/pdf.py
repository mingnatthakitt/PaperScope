from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

import fitz
from PIL import Image


class PDFValidationError(ValueError):
    """Raised when a file cannot be treated as a supported PDF."""


class PDFLimitError(PDFValidationError):
    """Raised when a valid PDF exceeds a configured size or page limit."""


@dataclass(frozen=True)
class PageText:
    page_number: int
    text: str
    blocks: list[str]


@dataclass(frozen=True)
class ChunkData:
    content: str
    page_start: int
    page_end: int
    section_title: str | None
    chunk_index: int
    chunk_type: str = "text"


@dataclass(frozen=True)
class RenderedPage:
    page_number: int
    path: Path
    width: int
    height: int


def sanitize_pdf_text(value: str) -> str:
    """Replace NULs emitted by some PDF text objects with a safe separator."""
    return value.replace("\x00", " ")


def sha256_file(path: Path, buffer_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(buffer_size):
            digest.update(chunk)
    return digest.hexdigest()


def validate_pdf(path: Path, max_bytes: int, max_pages: int) -> int:
    if path.stat().st_size > max_bytes:
        raise PDFLimitError(f"PDF exceeds the {max_bytes // (1024 * 1024)} MB limit")
    with path.open("rb") as file:
        if file.read(5) != b"%PDF-":
            raise PDFValidationError("Uploaded file is not a PDF")
    try:
        with fitz.open(path) as document:
            page_count = document.page_count
    except (fitz.FileDataError, ValueError) as exc:
        raise PDFValidationError("PDF could not be opened") from exc
    if page_count < 1:
        raise PDFValidationError("PDF has no pages")
    if page_count > max_pages:
        raise PDFLimitError(f"PDF exceeds the {max_pages}-page limit")
    return page_count


def _page_blocks(page: fitz.Page) -> list[str]:
    blocks: list[str] = []
    for block in page.get_text("blocks"):
        value = sanitize_pdf_text(str(block[4])).strip()
        if value:
            blocks.append(re.sub(r"\s+", " ", value))
    return blocks


def extract_pages(path: Path) -> list[PageText]:
    pages: list[PageText] = []
    with fitz.open(path) as document:
        for index, page in enumerate(document):
            blocks = _page_blocks(page)
            pages.append(PageText(index + 1, "\n\n".join(blocks), blocks))
    return pages


_SECTION_RE = re.compile(
    r"^(?:(?:\d+(?:\.\d+)*)\s+)?(?:abstract|introduction|background|related work|method|methodology|approach|experiments?|results?|discussion|limitations?|conclusion|references|appendix)(?:\s*[:\-–—]\s*)?$",
    re.IGNORECASE,
)


def _is_section_heading(value: str) -> bool:
    """Recognize short section labels without treating prose as a heading."""
    return len(value) <= 100 and bool(_SECTION_RE.fullmatch(value.strip()))


def detect_sections(pages: list[PageText]) -> list[tuple[str | None, int, int]]:
    sections: list[tuple[str | None, int, int]] = []
    current_title: str | None = None
    current_start = 1
    for page in pages:
        for block in page.blocks:
            first_line = block.splitlines()[0].strip()
            if _is_section_heading(first_line):
                if current_title is not None:
                    # Page-only section storage cannot represent two headings with
                    # different ranges on one page. Overlap that page deliberately,
                    # but never emit an impossible end-before-start range.
                    previous_end = max(current_start, page.page_number - 1)
                    sections.append((current_title, current_start, previous_end))
                elif page.page_number > current_start:
                    sections.append((current_title, current_start, page.page_number - 1))
                current_title = first_line
                current_start = page.page_number
    if pages:
        sections.append((current_title, current_start, pages[-1].page_number))
    return sections


def _section_for_page(page_number: int, sections: list[tuple[str | None, int, int]]) -> str | None:
    # When sections overlap on a page, the last heading is the best page-level
    # fallback. Block-level parsing below still preserves the transition itself.
    for title, start, end in reversed(sections):
        if start <= page_number <= end:
            return title
    return None


def chunk_pages(
    pages: list[PageText],
    sections: list[tuple[str | None, int, int]] | None = None,
    target_tokens: int = 900,
    overlap_tokens: int = 140,
) -> list[ChunkData]:
    """Group paragraph-like blocks while retaining page and section boundaries.

    Token counting is intentionally conservative and approximate. The generation models
    receive the original text; this function only controls retrieval unit size.
    """
    sections = sections or detect_sections(pages)
    units: list[tuple[str, int, str | None]] = []
    for page in pages:
        section_title = next(
            (title for title, start, _ in sections if start == page.page_number),
            _section_for_page(page.page_number, sections),
        )
        for block in page.blocks:
            first_line = block.splitlines()[0].strip()
            if _is_section_heading(first_line):
                section_title = first_line
            words = block.split()
            if words:
                units.append((block, page.page_number, section_title))

    chunks: list[ChunkData] = []
    current: list[tuple[str, int, str | None]] = []
    current_tokens = 0
    index = 0

    for unit in units:
        unit_tokens = len(unit[0].split())
        section_changed = bool(current and unit[2] != current[-1][2])
        if current and (section_changed or current_tokens + unit_tokens > target_tokens):
            chunks.append(
                ChunkData(
                    content="\n\n".join(item[0] for item in current),
                    page_start=current[0][1],
                    page_end=current[-1][1],
                    section_title=current[0][2],
                    chunk_index=index,
                )
            )
            index += 1
            if section_changed:
                current = []
                current_tokens = 0
                current.append(unit)
                current_tokens = unit_tokens
                continue
            overlap: list[tuple[str, int, str | None]] = []
            overlap_count = 0
            for previous in reversed(current):
                previous_tokens = len(previous[0].split())
                if overlap_count + previous_tokens > overlap_tokens:
                    break
                overlap.insert(0, previous)
                overlap_count += previous_tokens
            current = overlap
            current_tokens = overlap_count

        current.append(unit)
        current_tokens += unit_tokens

    if current:
        chunks.append(
            ChunkData(
                content="\n\n".join(item[0] for item in current),
                page_start=current[0][1],
                page_end=current[-1][1],
                section_title=current[0][2],
                chunk_index=index,
            )
        )
    return chunks


def render_pages(path: Path, output_dir: Path, dpi: int = 200) -> list[RenderedPage]:
    output_dir.mkdir(parents=True, exist_ok=True)
    scale = dpi / 72
    rendered: list[RenderedPage] = []
    with fitz.open(path) as document:
        for index, page in enumerate(document):
            pixmap = page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False)
            output_path = output_dir / f"{index + 1:03d}.webp"
            image = Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples)
            image.save(output_path, format="WEBP", quality=86, method=6)
            rendered.append(RenderedPage(index + 1, output_path, pixmap.width, pixmap.height))
    return rendered


def crop_normalized(page_path: Path, bbox: list[float], output_path: Path) -> Path:
    image = Image.open(page_path).convert("RGB")
    width, height = image.size
    left, top, crop_width, crop_height = bbox
    box = (
        max(0, int(left * width)),
        max(0, int(top * height)),
        min(width, int((left + crop_width) * width)),
        min(height, int((top + crop_height) * height)),
    )
    if box[2] <= box[0] or box[3] <= box[1]:
        raise ValueError("Visual crop has no area")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.crop(box).save(output_path, format="WEBP", quality=88, method=6)
    return output_path
