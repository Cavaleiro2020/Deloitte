"""
Gradio Web Interface for EcoGuide Chatbot

This module provides a web-based user interface for the RAG chatbot using Gradio.
Gradio is a Python library that makes it easy to create web UIs for ML models.

TODO: Complete the implementation of this Gradio interface to replace the terminal-based
      interaction in main.py. This will provide a better user experience with:
      - Chat history display
      - Source citation display
      - Product recommendations
      - File upload for new documents
      - Conversation export

REQUIREMENTS:
- Install gradio: pip install gradio
- Use the existing RAG components (FAISSIndex, LLM, EmbeddingsService)
- Maintain conversation history per session
- Display retrieved sources with each answer

GRADIO DOCUMENTATION:
- Main docs: https://www.gradio.app/docs
- Chat interface: https://www.gradio.app/docs/chatinterface
- Blocks API: https://www.gradio.app/docs/blocks
"""

import gradio as gr
from dotenv import load_dotenv
import os
import shutil
import csv
import re
from datetime import datetime
from pathlib import Path

# TODO: Import the necessary components from your RAG system
# Hint: You'll need FAISSIndex, LLM, and possibly Embeddings
from src.services.vectorial_db.faiss_index import FAISSIndex
from src.services.models.llm import LLM
from src.services.models.embeddings import Embeddings
from src.ingestion.ingest_files import ingest_files_data_folder
from src.ingestion.loaders.loader import Loader
from src.ingestion.chunking.token_chunking import text_to_chunks


# Load environment variables
load_dotenv(override=True)


# ============================================================================
# GLOBAL STATE MANAGEMENT
# ============================================================================

# TODO: Initialize your RAG components globally
# These should be initialized once when the app starts, not on every request
# 
# Example:
embeddings_service = Embeddings()
faiss_index = FAISSIndex(dimension=3072, embeddings=embeddings_service.get_embeddings)
llm = LLM()
#
# # Load existing index or ingest documents
try:
    faiss_index.load_index()
    print("✅ Loaded existing FAISS index")
except:
    print("📁 No existing index found. Ingesting documents...")
    ingest_files_data_folder(faiss_index)
    faiss_index.save_index()
    print("✅ Documents ingested and index saved")


# TODO: Initialize global variables for state management
conversation_histories = {}  # Dictionary to store conversation history per session
PRODUCTS_CACHE = None


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def format_sources(retrieved_chunks, num_sources=3):
    """
    Format retrieved document chunks as source citations.
    
    TODO: Implement source formatting to show which documents were used.
    
    Args:
        retrieved_chunks (list): List of text chunks retrieved from FAISS
        num_sources (int): Number of sources to display
    
    Returns:
        str: Formatted source citations in HTML or Markdown
    
    Example output:
        **Sources:**
        1. climate_report_2024.pdf (page 15)
        2. greenhouse_protocol_FAQ.html (section 3)
        3. sustainable_products.csv (row 45)
    
    HINT: Currently FAISS only returns text chunks without metadata.
          You need to modify faiss_index.py to store and return metadata
          (document name, page number, etc.) alongside chunks.
    """
    if not retrieved_chunks:
        return "**Sources used:** No sources retrieved."

    sources = ["**Sources used:**"]

    for i, chunk in enumerate(retrieved_chunks[:num_sources], start=1):
        if chunk is None:
            sources.append(f"{i}. _Empty source chunk_")
            continue

        if isinstance(chunk, dict):
            text = (
                chunk.get("text")
                or chunk.get("content")
                or chunk.get("chunk")
                or ""
            )

            source = (
                chunk.get("source")
                or chunk.get("filename")
                or chunk.get("document")
            )

            page = chunk.get("page")
            section = chunk.get("section")

            if source:
                citation = f"{i}. `{source}`"

                if page is not None:
                    citation += f", page {page}"
                elif section is not None:
                    citation += f", section {section}"

                sources.append(citation)
            else:
                preview = str(text).replace("\n", " ").strip()
                preview = " ".join(preview.split())

                if len(preview) > 250:
                    preview = preview[:250] + "..."

                sources.append(f"{i}. _{preview}_")

        else:
            preview = str(chunk).replace("\n", " ").strip()
            preview = " ".join(preview.split())

            if len(preview) > 250:
                preview = preview[:250] + "..."

            sources.append(f"{i}. _{preview}_")

    return "\n".join(sources)


def load_sustainable_products():
    """
    Load sustainable product rows from CSV once and cache them for future responses.
    """
    global PRODUCTS_CACHE

    if PRODUCTS_CACHE is not None:
        return PRODUCTS_CACHE

    product_paths = [
        Path("data") / "sustainable_products.csv",
        Path("sustainable_products.csv"),
    ]

    csv_path = next((path for path in product_paths if path.exists()), None)
    if csv_path is None:
        PRODUCTS_CACHE = []
        return PRODUCTS_CACHE

    try:
        with csv_path.open("r", encoding="utf-8-sig", newline="") as csv_file:
            PRODUCTS_CACHE = list(csv.DictReader(csv_file))
    except Exception:
        PRODUCTS_CACHE = []

    return PRODUCTS_CACHE


def extract_product_recommendations(response_text, user_message=None):
    """
    Extract product recommendations from the LLM response.
    
    TODO: Implement product recommendation extraction and formatting.
    
    Args:
        response_text (str): The LLM's response text
        user_message (str | None): The user's original message, when available
    
    Returns:
        str: Formatted product recommendations in HTML/Markdown, or empty string
    
    APPROACH 1: Parse LLM response for product mentions
    - Look for product names/IDs in the response
    - Match against sustainable_products.csv
    - Format as a nice product card
    
    APPROACH 2: Separate product search
    - Use an LLM to detect if user is asking for products
    - Provide the LLM with the user's query and the CSV data and ask for recommendations
    - Format results as product recommendations
    
    HINT: You'll need to load sustainable_products.csv and search it
          based on the user's query or LLM response.
    """
    if not response_text and not user_message:
        return ""

    user_text = str(user_message or "")
    response_text = str(response_text or "")
    combined_text = f"{user_text}\n{response_text}"
    lower_combined = combined_text.lower()
    lower_user = user_text.lower()

    product_keywords = [
        "product",
        "products",
        "recommend",
        "recommends",
        "recommended",
        "recommendation",
        "recommendations",
        "buy",
        "purchase",
        "swap",
        "swaps",
        "alternative",
        "sustainable alternative",
        "sustainable alternatives",
        "eco-friendly",
        "reusable",
        "plastic",
        "home",
        "packaging",
        "appliance",
        "appliances",
        "clothing",
        "cleaning",
        "toothbrush",
        "bottle",
        "bottles",
        "bag",
        "bags",
        "solar",
        "led",
    ]

    product_related = any(
        re.search(rf"(?<!\w){re.escape(keyword)}(?!\w)", lower_user)
        for keyword in product_keywords
    )

    if not product_related:
        return ""

    products = load_sustainable_products()
    if not products:
        return ""

    category_keywords = {
        "Reusable Items": [
            "reusable", "reuse", "plastic", "packaging", "bag", "bags",
            "bottle", "bottles", "straw", "straws", "wrap", "storage",
            "kitchen", "home", "container", "containers", "produce",
        ],
        "Energy Efficiency": [
            "home", "energy", "efficient", "efficiency", "electricity",
            "appliance", "appliances", "led", "bulb", "thermostat",
            "shower", "showerhead", "power", "carbon footprint",
        ],
        "Renewable Energy": [
            "solar", "renewable", "charger", "charging", "lantern",
            "energy", "radio", "bike", "bicycle", "outdoor",
        ],
        "Personal Care": [
            "personal", "care", "bathroom", "toothbrush", "toothpaste",
            "soap", "shampoo", "makeup", "comb", "hygiene",
        ],
        "Sustainable Fashion": [
            "clothing", "fashion", "shirt", "t-shirt", "shoes",
            "sneakers", "jacket", "textile", "recycled materials",
            "wear", "apparel",
        ],
    }

    stopwords = {
        "about", "after", "also", "alternatives", "and", "are", "can",
        "for", "from", "friendly", "have", "home", "into", "made",
        "more", "product", "products", "recommend", "sustainable",
        "that", "the", "their", "this", "with", "what", "when", "your",
    }
    query_tokens = {
        token
        for token in re.findall(r"[a-z0-9]+", lower_combined)
        if len(token) >= 4 and token not in stopwords
    }

    matches = []
    seen_names = set()

    for product in products:
        name = (
            product.get("Product Name")
            or product.get("product name")
            or product.get("name")
            or product.get("Name")
            or ""
        ).strip()
        category = (
            product.get("Category")
            or product.get("category")
            or ""
        ).strip()
        description = (
            product.get("Description")
            or product.get("description")
            or product.get("Why it fits")
            or ""
        ).strip()

        if not (name or category or description):
            continue

        dedupe_key = (name or description or category).lower()
        if dedupe_key in seen_names:
            continue

        score = 0
        searchable_fields = [name, category, description]

        name_lower = name.lower()
        category_lower = category.lower()
        description_lower = description.lower()

        if name_lower and name_lower in lower_combined:
            score += 8

        if category_lower and category_lower in lower_combined:
            score += 4

        for category_name, keywords in category_keywords.items():
            if category == category_name:
                for keyword in keywords:
                    if re.search(rf"(?<!\w){re.escape(keyword)}(?!\w)", lower_combined):
                        score += 2

        for field in searchable_fields:
            field_lower = field.lower()
            if field_lower and field_lower in lower_combined:
                score += 3 if field == name else 2

            for token in field_lower.replace("/", " ").replace("-", " ").split():
                token = re.sub(r"[^a-z0-9]", "", token)
                if len(token) >= 4 and token in query_tokens:
                    score += 1

        if "plastic" in lower_combined and "plastic" in description_lower:
            score += 3

        if "home" in lower_combined and category in {"Energy Efficiency", "Reusable Items", "Renewable Energy"}:
            score += 2

        if score > 0:
            matches.append((score, product))
            seen_names.add(dedupe_key)

    if not matches:
        return ""

    def review_score(product):
        try:
            return float(product.get("Review Score") or 0)
        except (TypeError, ValueError):
            return 0

    matches.sort(key=lambda item: (item[0], review_score(item[1])), reverse=True)

    lines = ["**Recommended sustainable products:**", ""]

    for index, (_, product) in enumerate(matches[:3], start=1):
        name = (
            product.get("Product Name")
            or product.get("product name")
            or product.get("name")
            or product.get("Name")
            or "Sustainable product"
        ).strip()
        category = (product.get("Category") or product.get("category") or "").strip()
        description = (
            product.get("Description")
            or product.get("description")
            or ""
        ).strip()
        review_score = (
            product.get("Review Score")
            or product.get("Sustainability score")
            or product.get("sustainability score")
            or product.get("score")
            or ""
        )
        price = product.get("Price ($)") or product.get("price") or product.get("Price") or ""
        countries = product.get("Countries") or product.get("countries") or ""

        lines.append(f"{index}. **{name}**")

        if category:
            lines.append(f"   - Category: {category}")

        if description:
            lines.append(f"   - Why it fits: {description}")

        if review_score:
            lines.append(f"   - Review score: {review_score}/5")

        if price:
            lines.append(f"   - Price: ${price}")

        if countries:
            lines.append(f"   - Available in: {countries}")

        lines.append("")

    return "\n".join(lines).strip()


def chatbot_response(message, history):
    """
    Main chatbot function that processes user input and returns a response.
    
    This function implements the RAG workflow:
    1. Retrieve relevant chunks from FAISS
    2. Augment the query with retrieved context
    3. Generate response using LLM
    4. Format the response with sources and recommendations
    
    Args:
        message (str): The user's input message
        history (list): Chat history in Gradio format [(user_msg, bot_msg), ...]
    
    Returns:
        str: The chatbot's response with sources and recommendations
    
    TODO: Implement the complete RAG pipeline here
    """
    if not message or not message.strip():
        return "Please write a question."

    try:
        llm_history = []

        for user_msg, bot_msg in history or []:
            if user_msg:
                llm_history.append({"role": "user", "content": user_msg})

            if bot_msg:
                llm_history.append({"role": "assistant", "content": bot_msg})

        retrieved_chunks = faiss_index.retrieve_chunks(message, num_chunks=5)
        context = "\n\n#####\n\n".join(retrieved_chunks)

        response = llm.get_response(llm_history, context, message)

        sources = format_sources(retrieved_chunks)
        formatted_response = f"{response}\n\n---\n{sources}"

        products = extract_product_recommendations(response, message)
        if products:
            formatted_response += f"\n\n{products}"

        return formatted_response

    except Exception as e:
        return (
            "❌ Error while generating response:\n\n"
            f"```text\n{type(e).__name__}: {e}\n```"
        )


def reset_conversation():
    """
    Reset the conversation history.
    
    TODO: Implement conversation reset functionality.
    
    Returns:
        tuple: Empty history and a message confirming reset
    """
    return [], "Conversation reset. Ask me anything about climate change or sustainable products."


def export_conversation(history):
    """
    Export the conversation history to a file.
    
    TODO: Implement conversation export functionality.
    
    Args:
        history (list): Chat history in Gradio format
    
    Returns:
        str: Path to the exported file, or error message
    
    HINT: You can export as:
    - Plain text (.txt)
    - JSON (.json) 
    - Markdown (.md)
    - PDF (requires reportlab or similar)
    """
    if not history:
        return None

    export_dir = Path("exports")
    export_dir.mkdir(exist_ok=True)

    exported_at = datetime.now()
    timestamp = exported_at.strftime("%Y%m%d_%H%M%S")
    export_path = export_dir / f"ecoguide_conversation_{timestamp}.md"

    lines = [
        "# EcoGuide AI Conversation Export",
        "",
        f"Exported at: {exported_at.isoformat(timespec='seconds')}",
        "",
        "---",
        "",
    ]

    if isinstance(history, list) and history and isinstance(history[0], dict):
        for msg in history:
            role = str(msg.get("role", "unknown")).capitalize()
            content = msg.get("content", "")

            if not content:
                continue

            lines.extend([
                f"## {role}",
                "",
                str(content),
                "",
                "---",
                "",
            ])
    else:
        for item in history:
            try:
                user_msg, bot_msg = item
            except (TypeError, ValueError):
                lines.extend([
                    "## Unknown",
                    "",
                    str(item),
                    "",
                    "---",
                    "",
                ])
                continue

            if user_msg:
                lines.extend([
                    "## User",
                    "",
                    str(user_msg),
                    "",
                    "---",
                    "",
                ])

            if bot_msg:
                lines.extend([
                    "## Assistant",
                    "",
                    str(bot_msg),
                    "",
                    "---",
                    "",
                ])

    export_path.write_text("\n".join(lines), encoding="utf-8")

    return str(export_path)


def upload_document(file):
    """
    Upload and ingest a new document into the RAG system.
    
    TODO: Implement document upload and ingestion.
    
    Args:
        file: File object from Gradio file upload component
    
    Returns:
        str: Status message about the upload
    
    WORKFLOW:
    1. Save the uploaded file to a temporary location
    2. Determine file type (PDF, HTML, DOCX, etc.)
    3. Load the document using appropriate loader
    4. Chunk the document text
    5. Generate embeddings and add to FAISS index
    6. Save the updated index
    7. Return success message
    
    HINT: You can reuse code from ingest_files.py
    """
    if file is None:
        return "Please select a file first."

    supported_extensions = {".pdf", ".html", ".htm", ".docx", ".pptx", ".csv", ".txt"}

    try:
        if isinstance(file, (str, Path)):
            source_path = Path(file)
            original_name = source_path.name
        elif isinstance(file, dict):
            source_path = Path(file.get("path") or file.get("name") or file.get("orig_name", ""))
            original_name = file.get("orig_name") or file.get("name") or source_path.name
        else:
            source_path = Path(file.name)
            original_name = Path(file.name).name

        if not source_path.exists():
            return f"Uploaded file could not be found: {source_path}"

        extension = source_path.suffix.lower()
        if extension not in supported_extensions:
            return (
                f"Unsupported file type: {extension}. "
                "Supported types are PDF, HTML, DOCX, PPTX, CSV and TXT."
            )

        upload_dir = Path("data") / "uploads"
        upload_dir.mkdir(parents=True, exist_ok=True)

        destination_path = upload_dir / Path(original_name).name
        if destination_path.exists():
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            destination_path = (
                upload_dir / f"{destination_path.stem}_{timestamp}{destination_path.suffix}"
            )

        shutil.copy2(source_path, destination_path)

        if extension in {".txt", ".csv"}:
            text = destination_path.read_text(encoding="utf-8", errors="replace")
        else:
            loader_extension = "html" if extension == ".htm" else extension.lstrip(".")
            loader = Loader(filepath=str(destination_path), extension=loader_extension)
            text = loader.extract_text()

        if not text or not text.strip():
            return f"Uploaded {destination_path.name}, but no text could be extracted."

        chunks = text_to_chunks(text)
        if not chunks:
            return f"Uploaded {destination_path.name}, but no chunks were created."

        faiss_index.ingest_text(text_chunks=chunks)
        faiss_index.save_index()

        return (
            f"Successfully uploaded and ingested {destination_path.name}. "
            f"Added {len(chunks)} chunks to the FAISS index."
        )

    except Exception as e:
        return f"Error while uploading document: {type(e).__name__}: {e}"


# ============================================================================
# GRADIO INTERFACE
# ============================================================================

def create_interface():
    """
    Create and configure the Gradio interface.
    
    TODO: Build a complete Gradio interface with multiple tabs/sections.
    
    RECOMMENDED STRUCTURE:
    
    Tab 1: Chat Interface
    - Chat history display (scrollable)
    - Message input box
    - Submit button
    - Clear conversation button
    - Export conversation button
    
    Tab 2: Document Management
    - File upload component
    - List of currently indexed documents
    - Re-index button
    
    Tab 3: Settings (Optional)
    - Number of chunks to retrieve (slider)
    - Temperature for LLM (slider)
    - Enable/disable source citations (checkbox)
    - Chunking strategy selection (dropdown)
    
    GRADIO COMPONENTS TO USE:
    - gr.ChatInterface: Pre-built chat interface (easiest)
    - gr.Blocks: Custom layout with more control
    - gr.Textbox: For message input
    - gr.Chatbot: For displaying chat history
    - gr.Button: For actions
    - gr.File: For file uploads
    - gr.Slider, gr.Checkbox, gr.Dropdown: For settings
    """
    
    global ECOGUIDE_CUSTOM_CSS

    ECOGUIDE_CUSTOM_CSS = """
    :root {
        --eco-sidebar: #073f31;
        --eco-sidebar-soft: #0d5a45;
        --eco-main: #f7faf8;
        --eco-primary: #10b981;
        --eco-primary-dark: #047857;
        --eco-text: #033b2d;
        --eco-muted: #5f736b;
        --eco-card: #ffffff;
        --eco-border: #dfe8e3;
    }

    body, .gradio-container {
        background: var(--eco-main) !important;
        font-family: Inter, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif !important;
        color: var(--eco-text);
    }

    .gradio-container {
        max-width: none !important;
        padding: 0 !important;
    }

    #app-shell {
        min-height: 100vh;
        gap: 0;
        background: var(--eco-main);
    }

    #sidebar {
        background: var(--eco-sidebar);
        min-width: 290px;
        max-width: 320px;
        padding: 24px 18px;
        color: #ffffff;
        border-right: 1px solid rgba(255, 255, 255, 0.08);
    }

    #sidebar .gr-form,
    #sidebar .block,
    #sidebar .wrap {
        background: transparent !important;
        border: 0 !important;
        box-shadow: none !important;
    }

    #brand-block h2 {
        color: #ffffff;
        margin: 0 0 4px;
        font-size: 22px;
        letter-spacing: 0;
    }

    #brand-block p,
    #sidebar-note,
    #sidebar-note p {
        color: rgba(255, 255, 255, 0.72);
        font-size: 13px;
        margin: 0;
    }

    .sidebar-section {
        margin-top: 22px;
        color: rgba(255, 255, 255, 0.72);
        font-size: 12px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }

    .sidebar-button button,
    .sidebar-prompt button {
        width: 100%;
        border-radius: 14px !important;
        border: 1px solid rgba(255, 255, 255, 0.14) !important;
        box-shadow: none !important;
        color: #ffffff !important;
    }

    .sidebar-button button {
        background: var(--eco-primary) !important;
        font-weight: 800 !important;
        min-height: 46px;
    }

    .sidebar-prompt button {
        background: rgba(255, 255, 255, 0.08) !important;
        justify-content: flex-start !important;
        text-align: left !important;
        min-height: 44px;
        font-size: 13px !important;
    }

    .sidebar-prompt button:hover,
    .sidebar-button button:hover {
        filter: brightness(1.08);
    }

    #sidebar .label-wrap span,
    #sidebar label,
    #sidebar .file-preview span {
        color: rgba(255, 255, 255, 0.82) !important;
    }

    #sidebar .file-preview,
    #sidebar .upload-container {
        background: rgba(255, 255, 255, 0.08) !important;
        border-color: rgba(255, 255, 255, 0.16) !important;
        border-radius: 14px !important;
    }

    #sidebar .markdown,
    #sidebar .markdown p {
        color: rgba(255, 255, 255, 0.8) !important;
        font-size: 13px;
    }

    #main-panel {
        padding: 26px 34px 28px;
        min-width: 0;
    }

    #top-header {
        background: rgba(255, 255, 255, 0.82);
        border: 1px solid var(--eco-border);
        border-radius: 22px;
        padding: 16px 20px;
        margin-bottom: 22px;
        box-shadow: 0 14px 40px rgba(8, 60, 45, 0.06);
    }

    #top-header .header-row {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 16px;
    }

    #top-header h1 {
        color: var(--eco-text);
        font-size: 24px;
        margin: 0;
        letter-spacing: 0;
    }

    #top-header .status {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        color: var(--eco-primary-dark);
        font-size: 13px;
        font-weight: 700;
        margin-top: 4px;
    }

    #top-header .dot {
        width: 9px;
        height: 9px;
        border-radius: 999px;
        background: var(--eco-primary);
        box-shadow: 0 0 0 5px rgba(16, 185, 129, 0.16);
    }

    #top-header .theme-pill {
        width: 42px;
        height: 42px;
        display: grid;
        place-items: center;
        border-radius: 999px;
        background: #ecfdf5;
        color: var(--eco-primary-dark);
        border: 1px solid #cdeee0;
        font-size: 18px;
    }

    #welcome-panel {
        text-align: center;
        max-width: 920px;
        margin: 18px auto 22px;
    }

    #welcome-panel .welcome-icon {
        width: 74px;
        height: 74px;
        border-radius: 24px;
        display: grid;
        place-items: center;
        margin: 0 auto 14px;
        background: linear-gradient(135deg, #10b981, #0f766e);
        color: #ffffff;
        font-size: 34px;
        box-shadow: 0 18px 42px rgba(16, 185, 129, 0.24);
    }

    #welcome-panel h2 {
        color: var(--eco-text);
        font-size: 34px;
        margin: 0 0 8px;
        letter-spacing: 0;
    }

    #welcome-panel p {
        color: var(--eco-muted);
        font-size: 16px;
        margin: 0 auto;
        max-width: 690px;
    }

    .prompt-card button {
        min-height: 112px;
        white-space: normal !important;
        text-align: left !important;
        align-items: flex-start !important;
        justify-content: flex-start !important;
        border-radius: 22px !important;
        background: var(--eco-card) !important;
        color: var(--eco-text) !important;
        border: 1px solid var(--eco-border) !important;
        box-shadow: 0 14px 32px rgba(8, 60, 45, 0.06) !important;
        font-weight: 700 !important;
        line-height: 1.35 !important;
        padding: 18px !important;
    }

    .prompt-card button:hover {
        border-color: #b7dfce !important;
        transform: translateY(-1px);
    }

    #chatbot {
        border-radius: 22px !important;
        border: 1px solid var(--eco-border) !important;
        background: #ffffff !important;
        box-shadow: 0 18px 46px rgba(8, 60, 45, 0.07);
        overflow: hidden;
    }

    #input-bar {
        margin-top: 16px;
        background: #ffffff;
        border: 1px solid var(--eco-border);
        border-radius: 24px;
        padding: 10px;
        box-shadow: 0 14px 36px rgba(8, 60, 45, 0.07);
    }

    #message-input textarea {
        border: 0 !important;
        box-shadow: none !important;
        background: transparent !important;
        font-size: 15px !important;
    }

    #send-button button {
        border-radius: 18px !important;
        background: var(--eco-primary) !important;
        color: #ffffff !important;
        border: 0 !important;
        font-weight: 800 !important;
        min-height: 48px;
    }

    #reset-status .markdown,
    #reset-status p {
        color: var(--eco-muted);
        font-size: 13px;
        margin: 4px 0 0;
    }

    @media (max-width: 900px) {
        #app-shell {
            flex-direction: column;
        }

        #sidebar {
            max-width: none;
            width: 100%;
        }

        #main-panel {
            padding: 20px;
        }
    }
    """

    def history_to_classic(history):
        classic_history = []

        if history and isinstance(history[0], dict):
            pending_user_message = None

            for chat_message in history:
                role = chat_message.get("role")
                content = chat_message.get("content")

                if not content:
                    continue

                if role == "user":
                    if pending_user_message is not None:
                        classic_history.append((pending_user_message, None))
                    pending_user_message = content
                elif role == "assistant":
                    if pending_user_message is not None:
                        classic_history.append((pending_user_message, content))
                        pending_user_message = None

            if pending_user_message is not None:
                classic_history.append((pending_user_message, None))
        else:
            classic_history = history or []

        return classic_history

    def submit_message(message, history):
        if not message or not message.strip():
            return history or [], ""

        history = history or []
        response = chatbot_response(message, history_to_classic(history))
        updated_history = [
            *history,
            {"role": "user", "content": message},
            {"role": "assistant", "content": response},
        ]
        return updated_history, ""

    def set_prompt(prompt):
        return prompt

    def reset_chat():
        empty_history, status = reset_conversation()
        return empty_history, status, ""

    with gr.Blocks(title="🌍 EcoGuide AI") as interface:
        with gr.Row(elem_id="app-shell"):
            with gr.Column(scale=0, min_width=290, elem_id="sidebar"):
                gr.HTML(
                    """
                    <div id="brand-block">
                        <h2>🌍 EcoGuide AI</h2>
                        <p>Climate intelligence workspace</p>
                    </div>
                    """
                )

                new_chat_btn = gr.Button("+ New Chat", elem_classes=["sidebar-button"])

                gr.Markdown("Suggested prompts", elem_classes=["sidebar-section"])
                side_prompt_1 = gr.Button(
                    "Climate change impacts on coastal cities",
                    elem_classes=["sidebar-prompt"],
                )
                side_prompt_2 = gr.Button(
                    "Sustainable packaging alternatives",
                    elem_classes=["sidebar-prompt"],
                )
                side_prompt_3 = gr.Button(
                    "Carbon footprint analysis",
                    elem_classes=["sidebar-prompt"],
                )

                gr.Markdown("Actions", elem_classes=["sidebar-section"])
                export_btn = gr.Button("Export conversation", elem_classes=["sidebar-prompt"])
                export_file = gr.File(label="Download Markdown export")

                gr.Markdown("Upload knowledge", elem_classes=["sidebar-section"])
                document_upload = gr.File(
                    label="📎 Upload PDF, HTML, DOCX, PPTX, CSV, or TXT",
                    file_types=[".pdf", ".html", ".htm", ".docx", ".pptx", ".csv", ".txt"],
                )
                upload_btn = gr.Button("Upload and ingest", elem_classes=["sidebar-button"])
                upload_status = gr.Markdown()

                gr.Markdown("⚙ Settings", elem_classes=["sidebar-section"])
                gr.Markdown(
                    "Local RAG mode · FAISS index · Azure OpenAI",
                    elem_id="sidebar-note",
                )

            with gr.Column(scale=1, elem_id="main-panel"):
                gr.HTML(
                    """
                    <div id="top-header">
                        <div class="header-row">
                            <div>
                                <h1>EcoGuide AI</h1>
                                <div class="status"><span class="dot"></span>Online and ready</div>
                            </div>
                            <div class="theme-pill">☼</div>
                        </div>
                    </div>
                    """
                )

                gr.HTML(
                    """
                    <div id="welcome-panel">
                        <div class="welcome-icon">🌱</div>
                        <h2>Welcome to EcoGuide AI</h2>
                        <p>Your AI assistant for sustainability, climate insights, and environmental intelligence</p>
                    </div>
                    """
                )

                with gr.Row():
                    prompt_1 = gr.Button(
                        "Climate Impact\nWhat are the latest IPCC findings on climate change?",
                        elem_classes=["prompt-card"],
                    )
                    prompt_2 = gr.Button(
                        "Sustainable Products\nSuggest eco-friendly alternatives to plastic packaging",
                        elem_classes=["prompt-card"],
                    )

                with gr.Row():
                    prompt_3 = gr.Button(
                        "Carbon Footprint\nHow can I reduce my carbon footprint at home?",
                        elem_classes=["prompt-card"],
                    )
                    prompt_4 = gr.Button(
                        "Green Energy\nCompare renewable energy sources for residential use",
                        elem_classes=["prompt-card"],
                    )

                chatbot = gr.Chatbot(
                    elem_id="chatbot",
                label="Conversation",
                render_markdown=True,
                    height=420,
                    placeholder="Ask about climate change, product swaps, emissions, and green energy.",
            )

                with gr.Row(elem_id="input-bar"):
                    message_input = gr.Textbox(
                        placeholder="Message EcoGuide AI...",
                        show_label=False,
                        lines=1,
                        scale=8,
                        elem_id="message-input",
                    )
                    send_btn = gr.Button("Send", scale=1, elem_id="send-button")

                reset_status = gr.Markdown(elem_id="reset-status")

            new_chat_btn.click(
                fn=reset_chat,
                inputs=None,
                outputs=[chatbot, reset_status, message_input],
                api_name="reset_conversation",
            )

            message_input.submit(
                fn=submit_message,
                inputs=[message_input, chatbot],
                outputs=[chatbot, message_input],
                api_name="submit_message",
            )
            send_btn.click(
                fn=submit_message,
                inputs=[message_input, chatbot],
                outputs=[chatbot, message_input],
            )

            prompt_map = [
                (side_prompt_1, "Climate change impacts on coastal cities"),
                (side_prompt_2, "Sustainable packaging alternatives"),
                (side_prompt_3, "Carbon footprint analysis"),
                (prompt_1, "What are the latest IPCC findings on climate change?"),
                (prompt_2, "Suggest eco-friendly alternatives to plastic packaging"),
                (prompt_3, "How can I reduce my carbon footprint at home?"),
                (prompt_4, "Compare renewable energy sources for residential use"),
            ]

            for prompt_button, prompt_text in prompt_map:
                prompt_button.click(
                    fn=lambda text=prompt_text: set_prompt(text),
                    inputs=None,
                    outputs=message_input,
                )

            export_btn.click(
                fn=export_conversation,
                inputs=chatbot,
                outputs=export_file,
                api_name="export_conversation",
            )

            upload_btn.click(
                fn=upload_document,
                inputs=document_upload,
                outputs=upload_status,
                api_name="upload_document",
            )
    
    return interface


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """
    Main function to launch the Gradio app.
    
    Launches the local Gradio app.
    """
    print("=" * 80)
    print("🌍 EcoGuide AI - Gradio Web Interface")
    print("=" * 80)
    print("Launching local Gradio app...")
    print("Open http://localhost:7860 in your browser.")

    interface = create_interface()

    interface.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        debug=True,
        show_error=True,
        theme=gr.themes.Soft(),
        css=ECOGUIDE_CUSTOM_CSS,
    )


if __name__ == "__main__":
    main()


# ============================================================================
# ADDITIONAL IMPLEMENTATION HINTS
# ============================================================================

"""
DEPLOYMENT OPTIONS:
===================

1. **Local Development**:
   python gradio_app.py
   # Access at http://localhost:7860

GRADIO FEATURES TO EXPLORE:
============================

1. **Theming**: Use gr.themes.Soft(), gr.themes.Base(), or create custom theme
2. **Authentication**: Add login with gr.Interface(..., auth=("username", "password"))
3. **Queue**: Handle multiple users with queue=True
4. **Flagging**: Let users flag good/bad responses for improvement
5. **State**: Use gr.State() to maintain user session data
6. **Layout**: Use gr.Row(), gr.Column() for custom layouts
7. **Markdown**: Rich formatting with gr.Markdown()
8. **Analytics**: Track usage with flagging or custom logging

TESTING CHECKLIST:
==================

Before deploying, test:
- [ ] Basic chat functionality works
- [ ] Source citations are displayed correctly
- [ ] Product recommendations appear when relevant
- [ ] Conversation can be cleared
- [ ] Conversation can be exported
- [ ] New documents can be uploaded and ingested
- [ ] Error handling (what if FAISS index is empty?)
- [ ] Mobile responsiveness (Gradio handles this automatically)
- [ ] Multiple concurrent users (use queue=True)


PERFORMANCE OPTIMIZATION:
=========================

1. **Lazy Loading**: Only load FAISS index when first needed
2. **Caching**: Cache frequently asked questions
3. **Async**: Use async/await for non-blocking operations
4. **Streaming**: Stream LLM responses for better UX (requires LLM streaming support)
5. **Batch Processing**: Process multiple user queries in batches if possible


SECURITY CONSIDERATIONS:
========================

1. **Input Validation**: Sanitize user inputs to prevent injection attacks
2. **Rate Limiting**: Prevent abuse by limiting requests per user
3. **File Upload**: Validate file types and sizes to prevent malicious uploads

Good luck with the implementation! 🚀
"""
