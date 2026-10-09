"""Turn statement PDFs into text the extractor can read.

Pages with a text layer are read directly. Pages without one (scans) go through
Tesseract OCR, so scanned and native statements take the same route afterwards.

The text is written to disk and committed. That keeps every model request
byte-for-byte reproducible, whichever machine or Tesseract version replays it.
"""

from __future__ import annotations

import json
from pathlib import Path

import pdfplumber
import pytesseract
from pdf2image import convert_from_path

MIN_TEXT_CHARS = 20  # fewer characters than this on a page means there is no usable text layer
OCR_DPI = 300


def pdf_to_text(pdf: Path) -> tuple[str, list[str]]:
    """Return the statement text and, per page, which route produced it."""
    pages, routes = [], []
    with pdfplumber.open(pdf) as doc:
        for number, page in enumerate(doc.pages, start=1):
            text = page.extract_text() or ""
            if len(text.strip()) >= MIN_TEXT_CHARS:
                routes.append("text_layer")
            else:
                image = convert_from_path(str(pdf), dpi=OCR_DPI, first_page=number, last_page=number)[0]
                text = pytesseract.image_to_string(image)
                routes.append("ocr")
            pages.append(f"[Page {number}]\n{text.strip()}")
    return "\n\n".join(pages) + "\n", routes


def prepare(data_dir: Path) -> dict[str, list[str]]:
    manifest = json.loads((data_dir / "manifest.json").read_text(encoding="utf-8"))
    out = data_dir / "text"
    out.mkdir(exist_ok=True)
    routes = {}
    for entry in manifest:
        text, page_routes = pdf_to_text(data_dir / "statements" / f"{entry['id']}.pdf")
        (out / f"{entry['id']}.txt").write_text(text, encoding="utf-8")
        routes[entry["id"]] = page_routes
    (out / "routes.json").write_text(json.dumps(routes, indent=2) + "\n", encoding="utf-8")
    return routes
