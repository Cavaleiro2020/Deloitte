"""Visual parsers: OCR and vision description adapters.

This module provides two main adapters:
- `ocr_text(image_path)` — returns raw OCR text (requires pytesseract)
- `vision_describe(image_path)` — attempts to call a vision model (placeholder)

The vision_describe function is implemented as a thin adapter that can be
extended to call Azure OpenAI vision/chat completions when configured.
"""

from pathlib import Path
import os
import json
import base64
import hashlib
import urllib.request
import urllib.error

try:
    from openai import AzureOpenAI
except Exception:
    AzureOpenAI = None

try:
    import pytesseract
    from PIL import Image
except Exception:
    pytesseract = None
    Image = None


def ocr_text(image_path: str) -> str:
    """Extract text from an image using pytesseract.

    Returns empty string if pytesseract is not available.
    """
    if pytesseract is None or Image is None:
        return ""  # OCR not available in this environment

    img = Image.open(image_path)
    try:
        text = pytesseract.image_to_string(img)
    except Exception:
        text = ""
    return text.strip()


def vision_describe(image_path: str) -> dict:
    """Describe an image using a vision-capable LLM.

    This is a placeholder that returns a minimal structure. Integrate with
    Azure OpenAI GPT-4o vision by calling the appropriate API and returning
    the parsed description and structured observations.

    Returns:
        {"description": str, "observations": list[str]}
    """
    # Simple caching based on image bytes
    img_path = Path(image_path)
    cache_dir = Path(".cache/vision")
    cache_dir.mkdir(parents=True, exist_ok=True)

    try:
        img_bytes = img_path.read_bytes()
    except Exception:
        return {"description": "", "observations": []}

    img_hash = hashlib.sha256(img_bytes).hexdigest()
    cache_file = cache_dir / f"{img_hash}.json"
    if cache_file.exists():
        try:
            return json.loads(cache_file.read_text(encoding="utf-8"))
        except Exception:
            pass

    # Prepare OCR fallback/context
    ocr_result = ocr_text(image_path)

    # Build a compact base64 representation to send inline
    b64 = base64.b64encode(img_bytes).decode("ascii")

    # Load Azure config from env (reuse same vars as LLM)
    azure_endpoint = os.getenv("AZURE_LLM_ENDPOINT")
    azure_deployment = os.getenv("AZURE_LLM_DEPLOYMENT_NAME")
    api_key = os.getenv("AZURE_LLM_API_KEY")
    api_version = os.getenv("AZURE_LLM_API_VERSION")

    # If Azure config missing, fall back to OCR-only description
    if not (azure_endpoint and azure_deployment and api_key and api_version):
        description = ocr_result or f"Image: {img_path.name}"
        result = {"description": description, "observations": []}
        try:
            cache_file.write_text(json.dumps(result), encoding="utf-8")
        except Exception:
            pass
        return result

    # Compose messages instructing the model to describe the image and return JSON
    SYSTEM_PROMPT = (
        "You are an image understanding assistant. Given an image and OCR text,"
        " produce a short plain-language description and a list of concise observations"
        " (e.g., 'contains a bar chart showing sales by year', 'table with 4 columns', 'photo of a person')."
        " Return output as a JSON object with keys: description (string) and observations (array of strings)."
    )

    # Azure vision accepts multimodal message parts with a base64 data URL.
    image_data_url = f"data:image/{img_path.suffix.lstrip('.').lower() or 'png'};base64,{b64}"
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": f"Filename: {img_path.name}\nOCR_TEXT:\n{ocr_result}"},
                {"type": "image_url", "image_url": {"url": image_data_url}},
            ],
        },
    ]

    try:
        url = f"{azure_endpoint.rstrip('/')}/openai/deployments/{azure_deployment}/chat/completions?api-version={api_version}"
        payload = json.dumps({
            "messages": messages,
            "temperature": 0.2,
            "max_completion_tokens": 500,
            "response_format": {"type": "json_object"},
        }).encode("utf-8")

        request = urllib.request.Request(
            url,
            data=payload,
            method="POST",
            headers={
                "Content-Type": "application/json",
                "api-key": api_key,
            },
        )

        with urllib.request.urlopen(request, timeout=60) as response:
            body = response.read().decode("utf-8")

        parsed_body = json.loads(body)
        text = parsed_body["choices"][0]["message"]["content"]

        # Try to extract JSON from the model output
        parsed = None
        try:
            parsed = json.loads(text)
        except Exception:
            # Attempt to find a JSON blob inside the text
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end != -1 and end > start:
                try:
                    parsed = json.loads(text[start:end+1])
                except Exception:
                    parsed = None

        if not parsed:
            parsed = {"description": text.strip(), "observations": []}

        # Ensure keys exist
        parsed.setdefault("description", "")
        parsed.setdefault("observations", [])

        # Cache the parsed result
        try:
            cache_file.write_text(json.dumps(parsed, ensure_ascii=False), encoding="utf-8")
        except Exception:
            pass

        return parsed

    except (urllib.error.HTTPError, urllib.error.URLError, KeyError, json.JSONDecodeError, Exception):
        # On any API error, fallback to OCR-based description
        description = ocr_result or f"Image: {img_path.name}"
        observations = []
        if ocr_result:
            observations.append("OCR text extracted from image")
        if img_path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}:
            observations.append(f"local image file: {img_path.name}")
        result = {"description": description, "observations": observations}
        try:
            cache_file.write_text(json.dumps(result), encoding="utf-8")
        except Exception:
            pass
        return result
