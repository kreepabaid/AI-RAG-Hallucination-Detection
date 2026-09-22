"""
llm_client.py
Uses huggingface_hub InferenceClient SDK with robust error handling,
dynamic API key/model switching, and a local extractive fallback mode.
"""

import os
import re
from typing import Optional, Tuple
from huggingface_hub import InferenceClient
from config import HF_API_KEY, LLM_MODEL, DEFAULT_LLM_MODEL


class HFInferenceClient:
    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        self.api_key = (api_key or os.getenv("HF_TOKEN") or os.getenv("HF_API_KEY") or HF_API_KEY or "").strip()
        self.model = model or LLM_MODEL or DEFAULT_LLM_MODEL
        self._init_client()

    def _init_client(self):
        if self.api_key and not self.api_key.startswith("hf_xxx"):
            self.client = InferenceClient(
                provider="auto",
                api_key=self.api_key,
            )
        else:
            self.client = None

    def set_api_key(self, api_key: str):
        """Update Hugging Face API key dynamically."""
        self.api_key = api_key.strip()
        self._init_client()

    def set_model(self, model: str):
        """Update active LLM model."""
        self.model = model.strip()

    def validate_connection(self) -> Tuple[bool, str]:
        """Test whether the current API key and model are operational."""
        if not self.api_key:
            return False, "No Hugging Face token provided."
        try:
            client = InferenceClient(api_key=self.api_key)
            resp = client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": "Respond with 'OK'."}],
                max_tokens=10,
            )
            return True, f"Connected successfully to {self.model}!"
        except Exception as e:
            err = str(e)
            if "401" in err or "expired" in err.lower() or "unauthorized" in err.lower():
                return False, "Token expired or invalid. Please check your token at https://huggingface.co/settings/tokens."
            elif "403" in err or "gated" in err.lower():
                return False, f"Model '{self.model}' is gated or restricted. Please accept its license on Hugging Face or switch to Qwen/Qwen2.5-7B-Instruct."
            return False, f"Inference API check failed: {err}"

    def _call(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.2
    ) -> str:
        if not self.client:
            raise ValueError(
                "Hugging Face API key not configured or invalid. "
                "Please enter your Hugging Face User Access Token in the sidebar or config.py."
            )

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=max_tokens,
                temperature=temperature,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            err_msg = str(e)
            if "expired" in err_msg.lower() or "401" in err_msg:
                raise PermissionError(
                    "Your Hugging Face API key is expired or invalid. "
                    "Please update it in the sidebar or config.py."
                ) from e
            elif "403" in err_msg:
                raise PermissionError(
                    f"Access forbidden to model '{self.model}'. You may need to request access on "
                    f"https://huggingface.co/{self.model} or switch models in settings."
                ) from e
            raise RuntimeError(f"Hugging Face inference error: {err_msg}") from e

    def answer_question(self, question: str, context: str) -> str:
        """Generate answer grounded strictly in retrieved context."""
        system = (
            "You are a helpful and factual assistant. "
            "Answer the question using ONLY the provided context. "
            "Be direct, concise, and factual. "
            "If the context does not contain enough information to answer, state clearly that the information is not available in the sources."
        )
        user = f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
        return self._call(system, user, max_tokens=400, temperature=0.2)

    def extract_claims(self, answer: str) -> str:
        """Break down generated answer into atomic verifiable factual claims."""
        system = (
            "You are a claim extraction assistant. "
            "Break the provided answer down into individual, atomic, verifiable factual claims. "
            "Output ONLY a numbered list of claims (e.g. 1. Claim one\\n2. Claim two). "
            "Do NOT include opinions, greetings, or explanations."
        )
        user = f"Answer:\n{answer}\n\nClaims:"
        return self._call(system, user, max_tokens=300, temperature=0.1)

    def correct_answer(
        self,
        question: str,
        original_answer: str,
        context: str,
        issues: str
    ) -> str:
        """Rewrite answer to remove hallucinated or unsupported statements."""
        system = (
            "You are a strict fact-checking assistant. "
            "Rewrite the answer using ONLY facts clearly supported by the provided context. "
            "Remove all statements flagged as unsupported or contradicted. "
            "Do not add any outside information."
        )
        user = (
            f"Context:\n{context}\n\n"
            f"Question: {question}\n\n"
            f"Original Answer (contained hallucinations):\n{original_answer}\n\n"
            f"Issues found:\n{issues}\n\n"
            f"Write the corrected, 100% source-grounded answer:"
        )
        return self._call(system, user, max_tokens=400, temperature=0.1)

    # ── Fallback Extractive Mode (when no API key is provided) ─────────────
    def fallback_answer_question(self, question: str, context: str) -> str:
        """Extractive fallback for local testing without an active HF key."""
        sentences = [s.strip() for s in re.split(r"(?<=[.?!])\s+", context) if len(s.strip()) > 15]
        q_words = set(re.findall(r"\w+", question.lower())) - {"what", "when", "where", "who", "which", "how", "the", "is", "are", "was", "did"}
        scored = []
        for s in sentences:
            s_words = set(re.findall(r"\w+", s.lower()))
            overlap = len(q_words & s_words)
            if overlap > 0:
                scored.append((overlap, s))
        scored.sort(key=lambda x: x[0], reverse=True)
        top = [s for _, s in scored[:3]]
        if not top:
            return "Based on the retrieved context, there is insufficient direct information to answer this question."
        return " ".join(top)

    def fallback_extract_claims(self, answer: str) -> str:
        """Extractive claim splitter fallback."""
        sentences = [s.strip() for s in re.split(r"(?<=[.?!])\s+", answer) if len(s.strip()) > 10]
        return "\n".join(f"{i+1}. {s}" for i, s in enumerate(sentences))
