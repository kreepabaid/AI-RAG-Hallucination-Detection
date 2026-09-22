"""
rag_pipeline.py
Orchestrates: document loading → embedding → retrieval → LLM answer → hallucination check → correction.
"""

from typing import List, Dict, Optional
from src.document_loader import load_and_chunk, chunk_raw_text
from src.vector_store import VectorStore
from src.llm_client import HFInferenceClient
from src.hallucination_checker import HallucinationChecker
from config import TOP_K


class HallucinationAwareRAG:
    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        self.vector_store = VectorStore()
        self.llm = HFInferenceClient(api_key=api_key, model=model)
        self.checker = HallucinationChecker()
        self.sources_loaded: List[str] = []

    def set_api_key(self, api_key: str):
        """Update LLM API key."""
        self.llm.set_api_key(api_key)

    def set_model(self, model: str):
        """Update LLM model."""
        self.llm.set_model(model)

    def add_source(self, source: str, original_name: Optional[str] = None) -> int:
        """Load a file or URL, chunk it, embed and index it. Returns chunk count."""
        chunks = load_and_chunk(source, original_name=original_name)
        if not chunks:
            return 0
        self.vector_store.add_chunks(chunks)
        display_name = original_name or source
        if display_name not in self.sources_loaded:
            self.sources_loaded.append(display_name)
        return len(chunks)

    def add_raw_text(self, text: str, source_name: str = "Pasted Text") -> int:
        """Add direct text input into the vector store."""
        chunks = chunk_raw_text(text, source_name=source_name)
        if not chunks:
            return 0
        self.vector_store.add_chunks(chunks)
        if source_name not in self.sources_loaded:
            self.sources_loaded.append(source_name)
        return len(chunks)

    def clear_sources(self):
        """Wipe all loaded sources from memory."""
        self.vector_store.clear()
        self.sources_loaded = []

    def query(
        self,
        question: str,
        top_k: int = TOP_K,
        allow_fallback: bool = True
    ) -> Dict:
        """
        Full pipeline for one question:
          1. Retrieve relevant chunks from vector store
          2. LLM generates answer from context
          3. LLM breaks answer into individual claims
          4. NLI model verifies each claim against retrieved sources
          5. If hallucinations found → LLM generates corrected answer
        """
        if self.vector_store.total_chunks == 0:
            return {"error": "No sources loaded. Please upload at least one document or paste text first."}

        # Step 1 – Retrieve
        hits = self.vector_store.search(question, top_k=top_k)
        if not hits:
            return {"error": "No relevant chunks found for this question."}

        retrieved_chunks = [chunk for chunk, _ in hits]
        retrieved_evidence = [
            {
                "chunk_id": chunk.get("chunk_id", i + 1),
                "source": chunk.get("source", "Unknown"),
                "text": chunk.get("text", ""),
                "similarity": round(float(score), 4),
                "similarity_pct": round(float(score) * 100, 1),
            }
            for i, (chunk, score) in enumerate(hits)
        ]
        context = "\n\n---\n\n".join([chunk["text"] for chunk in retrieved_chunks])
        sources_used = list(dict.fromkeys(chunk["source"] for chunk in retrieved_chunks))

        # Step 2 – Generate answer
        is_fallback = False
        try:
            answer = self.llm.answer_question(question, context)
        except Exception as e:
            if allow_fallback:
                is_fallback = True
                answer = self.llm.fallback_answer_question(question, context)
            else:
                raise e

        # Step 3 – Extract claims
        if is_fallback:
            claims_text = self.llm.fallback_extract_claims(answer)
        else:
            try:
                claims_text = self.llm.extract_claims(answer)
            except Exception:
                claims_text = self.llm.fallback_extract_claims(answer)

        # Step 4 – Verify claims using NLI model across retrieved chunks
        check = self.checker.check(claims_text, context=context, chunks=retrieved_chunks)

        # Step 5 – Correct if needed
        corrected_answer = None
        if check["hallucination_score"] > 0:
            if is_fallback:
                # Local fallback correction: keep only supported claims
                supported_claims = [c["claim"] for c in check["claims"] if c["verdict"] == "supported"]
                if supported_claims:
                    corrected_answer = " ".join(supported_claims)
                else:
                    corrected_answer = "The retrieved sources do not provide sufficient verified evidence to answer this question."
            else:
                try:
                    corrected_answer = self.llm.correct_answer(
                        question=question,
                        original_answer=answer,
                        context=context,
                        issues=check["issues"],
                    )
                except Exception:
                    supported_claims = [c["claim"] for c in check["claims"] if c["verdict"] == "supported"]
                    corrected_answer = " ".join(supported_claims) if supported_claims else "Information not verified in source context."

        return {
            "question": question,
            "answer": answer,
            "corrected_answer": corrected_answer,
            "needs_correction": check["hallucination_score"] > 0,
            "hallucination_score": check["hallucination_score"],
            "check_summary": check["summary"],
            "claims": check["claims"],
            "context_used": context,
            "sources_referenced": sources_used,
            "retrieved_evidence": retrieved_evidence,
            "is_fallback": is_fallback,
        }
