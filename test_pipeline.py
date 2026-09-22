"""
test_pipeline.py
Automated end-to-end verification of all core modules:
- Document Loader
- Vector Store (FAISS + SentenceTransformers)
- NLI Hallucination Checker (CrossEncoder)
- Full RAG Pipeline
"""

import sys
import io
from pathlib import Path

# Ensure UTF-8 output on Windows console
if sys.stdout.encoding != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.document_loader import load_and_chunk, chunk_raw_text
from src.vector_store import VectorStore
from src.hallucination_checker import HallucinationChecker
from src.rag_pipeline import HallucinationAwareRAG


def test_document_loader():
    print("\n--- Testing Document Loader ---")
    sample_text = (
        "The Apollo 11 mission landed the first two humans on the Moon. "
        "Commander Neil Armstrong and lunar module pilot Buzz Aldrin landed the Apollo Lunar Module Eagle on July 20, 1969. "
        "Armstrong stepped onto the lunar surface six hours and 39 minutes later on July 21. "
        "Aldrin joined him 19 minutes later. They spent about two and a quarter hours together outside the spacecraft."
    )
    chunks = chunk_raw_text(sample_text, source_name="Apollo 11 Document", chunk_size=30, overlap=10)
    assert len(chunks) > 0, "Failed to create chunks from raw text!"
    print(f"SUCCESS: Created {len(chunks)} chunks with proper metadata.")
    return chunks


def test_vector_store(chunks):
    print("\n--- Testing Vector Store & FAISS ---")
    store = VectorStore()
    store.add_chunks(chunks)
    assert store.total_chunks == len(chunks), "Chunk count mismatch in vector store!"

    hits = store.search("When did Neil Armstrong step on the moon?", top_k=2)
    assert len(hits) > 0, "No hits retrieved from vector store!"
    print(f"SUCCESS: Retrieved {len(hits)} hits. Top hit score: {hits[0][1]:.4f}")
    return store


def test_hallucination_checker():
    print("\n--- Testing Hallucination Checker (NLI) ---")
    checker = HallucinationChecker()

    premise = (
        "Commander Neil Armstrong and lunar module pilot Buzz Aldrin landed the Apollo Lunar Module Eagle on July 20, 1969. "
        "Neil Armstrong stepped onto the lunar surface on July 21, 1969."
    )

    claims_text = (
        "1. Neil Armstrong stepped onto the lunar surface in July 1969.\n"
        "2. Neil Armstrong was accompanied by Buzz Aldrin.\n"
        "3. Neil Armstrong landed on Mars in 1999."
    )

    result = checker.check(claims_text, context=premise)
    print("Checked summary:", result["summary"].encode("ascii", "replace").decode("ascii"))
    print("Hallucination score:", result["hallucination_score"], "%")

    claims = result["claims"]
    assert len(claims) == 3, f"Expected 3 claims, got {len(claims)}"

    # Claim 1 should be supported
    assert claims[0]["verdict"] == "supported", f"Claim 1 expected supported, got {claims[0]['verdict']}"
    # Claim 2 should be supported
    assert claims[1]["verdict"] == "supported", f"Claim 2 expected supported, got {claims[1]['verdict']}"
    # Claim 3 should be contradicted or unsupported
    assert claims[2]["verdict"] in ("contradicted", "unsupported"), f"Claim 3 should not be supported! Got {claims[2]['verdict']}"

    print(f"Claim 1: '{claims[0]['claim']}' -> {claims[0]['verdict']} ({claims[0]['confidence']}%)")
    print(f"Claim 2: '{claims[1]['claim']}' -> {claims[1]['verdict']} ({claims[1]['confidence']}%)")
    print(f"Claim 3: '{claims[2]['claim']}' -> {claims[2]['verdict']} ({claims[2]['confidence']}%)")
    print("SUCCESS: NLI Hallucination checker correctly distinguished grounded facts from hallucinations!")


def test_rag_pipeline():
    print("\n--- Testing Full End-to-End RAG Pipeline ---")
    rag = HallucinationAwareRAG()
    sample_doc = (
        "Python was conceived in the late 1980s by Guido van Rossum at Centrum Wiskunde & Informatica (CWI) in the Netherlands. "
        "Python 2.0 was released on 16 October 2000, and Python 3.0 was released on 3 December 2008."
    )
    rag.add_raw_text(sample_doc, source_name="Python History")
    res = rag.query("Who created Python and when was it conceived?", top_k=2)

    assert "answer" in res, "No answer generated!"
    assert "hallucination_score" in res, "No hallucination score!"
    assert "claims" in res, "No claims returned!"

    print("Pipeline Question:", res["question"])
    print("Pipeline Answer:", res["answer"])
    print("Pipeline Score:", res["hallucination_score"], "%")
    print(f"Claims verified: {len(res['claims'])}")
    print("SUCCESS: Full RAG pipeline executed successfully!")


if __name__ == "__main__":
    print("=== STARTING AI HALLUCINATION-AWARE RAG TESTS ===")
    chunks = test_document_loader()
    test_vector_store(chunks)
    test_hallucination_checker()
    test_rag_pipeline()
    print("\nALL TESTS PASSED SUCCESSFULLY!")
