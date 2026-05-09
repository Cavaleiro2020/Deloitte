"""Utilities to extract page images from PDFs for visual analysis.

This module uses `pdf2image` when available to rasterize PDF pages to images.
Each produced image is saved to a temporary directory and returned with page metadata.

This is intentionally lightweight: it rasterizes full pages so the visual parser
can decide how to analyze charts/figures. For higher-precision figure bounding
box detection, extend this module to run OpenCV-based figure detection.
"""

from pathlib import Path
import tempfile
import os
from typing import List, Dict

try:
    from pdf2image import convert_from_path
except Exception:
    convert_from_path = None


def rasterize_pdf_pages(pdf_path: str, output_dir: str | None = None) -> List[Dict]:
    """Rasterize each PDF page to an image file.

    Returns a list of dicts: {"page": int, "image_path": str}
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    if convert_from_path is None:
        raise RuntimeError("pdf2image not available. Install with 'pip install pdf2image' and ensure poppler is installed.")

    out_dir = Path(output_dir) if output_dir else Path(tempfile.mkdtemp(prefix="multimodal_"))
    out_dir.mkdir(parents=True, exist_ok=True)

    images = convert_from_path(str(pdf_path))
    results = []
    for i, img in enumerate(images, start=1):
        out_path = out_dir / f"{pdf_path.stem}_page_{i}.png"
        img.save(out_path)
        results.append({"page": i, "image_path": str(out_path)})

    return results
