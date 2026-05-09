# Multimodal RAG Implementation Plan

## Goal

Extract information from images/charts embedded in PDFs and index those visual-derived text chunks alongside document text in FAISS. Make visual chunks include metadata (file, page, figure id) so the LLM can cite visual sources.

## What I implemented (scaffold)

- `src/ingestion/image_extractor.py`: rasterize PDF pages to image files (uses `pdf2image`).
- `src/ingestion/visual_parsers.py`: OCR (`pytesseract`) adapter and placeholder `vision_describe()` function for GPT-4o vision integration.
- `src/ingestion/multimodal_ingest.py`: runner that converts page images into OCR and vision text chunks with metadata and calls `FAISSIndex.ingest_text`.
- `src/services/vectorial_db/faiss_index.py` (updated): store parallel `metadata_list` and return `(text, metadata)` tuples from `retrieve_chunks()`.

## Next steps to complete

1. Integrate `vision_describe()` with your Azure GPT-4o vision endpoint to obtain richer chart descriptions and structured observations.
2. Improve image/figure detection (OpenCV) to extract per-figure bounding boxes instead of whole-page images.
3. Add table extraction (`camelot` or `tabula`) for PDFs with native tables.
4. Wire visual chunk citations into `main.py` and `gradio_app.py` so the UI can show thumbnails and cite figures.

## Frontend wiring TODO

- Add an upload path in `gradio_app.py` for PDF/image files that triggers multimodal ingestion.
- Add a visual results panel that shows `description`, `observations`, and source metadata for each image chunk.
- Render figure thumbnails or page previews next to retrieved answers when visual metadata is present.
- Add a toggle for `use_vision` so users can choose OCR-only or Azure vision ingestion.
- Surface ingestion status and errors in the UI so users can tell whether OCR, Azure vision, or caching was used.
- Add a small source-citation component that links the answer back to file name and page number.

## How to run a quick test (local)

1. Install optional deps:

```bash
pip install pdf2image pillow pytesseract
# On Windows/WSL/Mac: install poppler and tesseract system packages
```

2. In Python (example):

```py
from src.services.models.embeddings import Embeddings
from src.services.vectorial_db.faiss_index import FAISSIndex
from src.ingestion.multimodal_ingest import multimodal_ingest

emb = Embeddings()
index = FAISSIndex(embeddings=emb.get_embeddings, dimension=3072)
multimodal_ingest('data/climate_change_overview_en.pdf', index, use_vision=False)
index.save_index()
```

## Risks & notes

- Vision API calls are costly; cache outputs and make ingestion optional.
- OCR quality varies with PDF rasterization — test with your sample PDFs.
- Start with page-level processing; add figure detection later if needed.
