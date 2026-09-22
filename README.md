# 🔍 Hallucination-Aware RAG System

> An intelligent Retrieval-Augmented Generation (RAG) system that detects, verifies, and corrects hallucinated claims using retrieved evidence and Natural Language Inference (NLI).

## 📌 Overview

Large Language Models (LLMs) can generate fluent but factually incorrect information, commonly known as **hallucinations**.

This project implements a **Hallucination-Aware Retrieval-Augmented Generation pipeline** that does not simply generate an answer. Instead, it:

1. Retrieves relevant information from user-provided sources.
2. Generates an answer using the retrieved context.
3. Breaks the generated answer into individual factual claims.
4. Verifies each claim against the retrieved evidence using an NLI model.
5. Classifies claims as **Supported, Unsupported, or Contradicted**.
6. Calculates a hallucination risk score.
7. Automatically generates a corrected, source-grounded response when required.

---

## 🎯 Problem Statement

Traditional RAG systems improve factuality by supplying external context to an LLM, but the generated response can still contain claims that are unsupported or contradictory to the retrieved evidence.

This project addresses this problem by adding a **post-generation verification layer** based on Natural Language Inference.

The system therefore follows:

```text
Retrieve → Generate → Extract Claims → Verify → Correct