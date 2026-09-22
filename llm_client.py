"""
Root shim for llm_client.py (forwards to src.llm_client)
"""

from src.llm_client import HFInferenceClient

__all__ = ["HFInferenceClient"]