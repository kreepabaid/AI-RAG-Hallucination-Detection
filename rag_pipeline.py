"""
Root shim for rag_pipeline.py (forwards to src.rag_pipeline)
"""

from src.rag_pipeline import HallucinationAwareRAG

__all__ = ["HallucinationAwareRAG"]