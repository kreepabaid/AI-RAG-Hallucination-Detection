"""
document_loader.py
Loads and chunks PDF, TXT, DOCX, CSV, MD files, web URLs, and direct text.
"""

import re
import os
import requests
from typing import List, Dict, Optional
from bs4 import BeautifulSoup


def load_txt(file_path: str) -> str:
    """Read a text or markdown file."""
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def load_pdf(file_path: str) -> str:
    """Extract text from a PDF file using pypdf."""
    from pypdf import PdfReader
    reader = PdfReader(file_path)
    text = ""
    for page_idx, page in enumerate(reader.pages):
        page_text = page.extract_text() or ""
        if page_text.strip():
            text += page_text + "\n"
    return text.strip()


def load_docx(file_path: str) -> str:
    """Extract text from a DOCX document."""
    from docx import Document
    doc = Document(file_path)
    return "\n".join([p.text for p in doc.paragraphs if p.text.strip()])


def load_csv(file_path: str) -> str:
    """Extract formatted text from a CSV file."""
    import csv
    lines = []
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        reader = csv.reader(f)
        for row in reader:
            if any(cell.strip() for cell in row):
                lines.append(" | ".join(row))
    return "\n".join(lines)


def load_url(url: str) -> str:
    """Fetch and extract clean readable text from a web URL."""
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    response = requests.get(url, headers=headers, timeout=15)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header", "aside", "noscript", "svg"]):
        tag.decompose()
    return soup.get_text(separator="\n", strip=True)


def load_document(source: str, original_name: Optional[str] = None) -> Dict:
    """
    Auto-detect source type and load it.
    Returns dict with {"text": str, "source": str, "type": str}.
    """
    display_name = original_name or os.path.basename(source)

    if source.startswith("http://") or source.startswith("https://"):
        return {"text": load_url(source), "source": original_name or source, "type": "url"}

    lower = source.lower()
    if lower.endswith(".pdf"):
        text = load_pdf(source)
        doc_type = "pdf"
    elif lower.endswith(".docx"):
        text = load_docx(source)
        doc_type = "docx"
    elif lower.endswith(".csv"):
        text = load_csv(source)
        doc_type = "csv"
    else:
        text = load_txt(source)
        doc_type = "txt"

    return {"text": text, "source": display_name, "type": doc_type}


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 100) -> List[str]:
    """Split text into overlapping word-based chunks safely."""
    if not text or not text.strip():
        return []

    # Clean whitespace
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    words = text.split()

    if not words:
        return []

    # Safeguard overlap
    if overlap >= chunk_size:
        overlap = max(0, chunk_size - 1)

    step = max(1, chunk_size - overlap)
    chunks = []
    start = 0

    while start < len(words):
        chunk = " ".join(words[start: start + chunk_size])
        if chunk.strip():
            chunks.append(chunk)
        start += step

    return chunks


def load_and_chunk(
    source: str,
    chunk_size: int = 500,
    overlap: int = 100,
    original_name: Optional[str] = None
) -> List[Dict]:
    """Load a source and return a list of chunk dicts with text + metadata."""
    doc = load_document(source, original_name=original_name)
    chunks = chunk_text(doc["text"], chunk_size, overlap)
    return [
        {
            "text": chunk,
            "source": doc["source"],
            "type": doc["type"],
            "chunk_id": i,
        }
        for i, chunk in enumerate(chunks)
    ]


def chunk_raw_text(
    text: str,
    source_name: str = "Direct Input",
    chunk_size: int = 500,
    overlap: int = 100
) -> List[Dict]:
    """Helper to chunk raw text directly provided by user."""
    chunks = chunk_text(text, chunk_size, overlap)
    return [
        {
            "text": chunk,
            "source": source_name,
            "type": "text",
            "chunk_id": i,
        }
        for i, chunk in enumerate(chunks)
    ]
