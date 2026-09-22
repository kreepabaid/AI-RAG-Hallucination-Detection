"""
Root shim for document_loader.py (forwards to src.document_loader)
"""

from src.document_loader import (
    load_txt,
    load_pdf,
    load_docx,
    load_csv,
    load_url,
    load_document,
    chunk_text,
    load_and_chunk,
    chunk_raw_text,
)

__all__ = [
    "load_txt",
    "load_pdf",
    "load_docx",
    "load_csv",
    "load_url",
    "load_document",
    "chunk_text",
    "load_and_chunk",
    "chunk_raw_text",
]