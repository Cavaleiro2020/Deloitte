# CSV Product Recommendations Implementation

## What changed

- Added a new CSV loader at `src/ingestion/loaders/loaderCSV.py` to parse `data/sustainable_products.csv` into normalized product records.
- Registered CSV support in `src/ingestion/loaders/loader.py` so the factory can construct the loader when needed.
- Added `src/services/product_recommender.py` to score and rank sustainable products with simple rule-based matching.
- Updated `main.py` to route product-intent questions to the CSV recommender before the normal FAISS + LLM RAG path.
- Created this documentation file to capture the implementation and the reason for the design choices.

## Why

- The product catalog is structured data, so it is better handled as a dedicated recommendation flow than mixed into the document FAISS index.
- Keeping product lookup separate avoids polluting the climate document embeddings and keeps retrieval quality stable.
- A rule-based recommender is predictable, easy to explain, and fast enough for the bootcamp challenge.

## Notes

- The existing document ingestion flow still skips CSV files, which keeps the climate knowledge base unchanged.
- The recommender returns markdown-formatted product tables, which can be reused in a future Gradio interface without changing the scoring logic.

## Next Steps

1. Wire the product response format into `gradio_app.py` so the Gradio callback can show the chat answer and the product table separately, instead of only returning plain text.
2. Add source citation support for the normal climate RAG answers, so retrieved document chunks can be shown with document names and page numbers instead of the current placeholder logic.

## What I Mean By Step 2

"Source citation support" means the non-product chatbot path should show where its answer came from. Right now the climate RAG path retrieves chunks from FAISS, but those chunks do not yet carry metadata like file name or page number. The next change would store that metadata during ingestion, return it with retrieval results, and then format it in the final answer as citations such as `Source: climate_report.pdf, page 5`.