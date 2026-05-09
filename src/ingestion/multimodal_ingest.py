"""Multimodal ingestion helper: extract visuals and index them into FAISS.

This script rasterizes PDF pages, runs OCR and optional vision descriptions,
converts results into text chunks with metadata, and calls `FAISSIndex.ingest_text`.

Usage: import and call `multimodal_ingest(pdf_path, index)` from a runner.
"""

from src.ingestion.image_extractor import rasterize_pdf_pages
from src.ingestion.visual_parsers import ocr_text, vision_describe
from src.services.vectorial_db.faiss_index import FAISSIndex
from typing import List
from pathlib import Path


def multimodal_ingest(pdf_path: str, index: FAISSIndex, use_vision: bool = True):
    pages = rasterize_pdf_pages(pdf_path)
    chunks = []

    for p in pages:
        image_path = p["image_path"]
        page_num = p["page"]

        # OCR text
        text = ocr_text(image_path)
        if text:
            chunks.append({
                "text": f"[OCR page {page_num}]\n" + text,
                "metadata": {"source_file": Path(pdf_path).name, "page": page_num, "type": "ocr", "image_path": image_path},
            })

        # Vision description (if available / desired)
        if use_vision:
            desc = vision_describe(image_path)
            if desc and desc.get("description"):
                chunks.append({
                    "text": f"[Vision page {page_num}]\n" + desc.get("description", ""),
                    "metadata": {"source_file": Path(pdf_path).name, "page": page_num, "type": "vision", "image_path": image_path, "observations": desc.get("observations", [])},
                })

    # Ingest all visual chunks into FAISS
    if chunks:
        index.ingest_text(text_chunks=chunks)
        return True

    return False
