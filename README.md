# ⚡ VerifAI – Advanced Hallucination-Aware RAG Platform

<p align="center">
  <b>Verify Every Claim. Eliminate Unsupported Answers. Build Trust in AI.</b>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue" />
  <img src="https://img.shields.io/badge/Framework-Streamlit-FF4B4B" />
  <img src="https://img.shields.io/badge/Vector%20Search-FAISS-green" />
  <img src="https://img.shields.io/badge/Retrieval-BM25-orange" />
  <img src="https://img.shields.io/badge/AI-Hugging%20Face-yellow" />
  <img src="https://img.shields.io/badge/LLM-Groq-purple" />
  <img src="https://img.shields.io/badge/License-MIT-lightgrey" />
</p>

---

## 📖 Overview

**VerifAI** is an advanced Hallucination-Aware Retrieval-Augmented Generation (RAG) platform designed to improve the reliability and factual accuracy of AI-generated responses.

Traditional RAG systems retrieve relevant documents and generate answers but may still produce unsupported or incorrect information.

VerifAI addresses this limitation by combining:

- **Dense Semantic Retrieval:** FAISS-based vector search.
- **Sparse Lexical Retrieval:** BM25 keyword matching.
- **Hybrid Search:** Reciprocal Rank Fusion (RRF).
- **NLI-Based Verification:** Cross-encoder models for claim-level fact checking.
- **Automated Guardrails:** Policy-based hallucination detection and correction.

Every generated answer is broken down into individual claims and evaluated against retrieved source evidence.

The platform provides color-coded verification, evidence citations, groundedness scores, hallucination risk scores, and automated answer correction.

The goal is simple: **Make AI-generated answers more transparent, explainable, and trustworthy.**

---

## ✨ Key Features

### 🚀 1. Multi-Provider LLM Integration

VerifAI supports multiple language model providers, allowing users to select their preferred inference engine.

| Provider | Description |
|---|---|
| Groq | Fast cloud-based LLM inference |
| Google Gemini | Google's generative AI models |
| Ollama | Local and private LLM execution |
| OpenAI | OpenAI-compatible model integration |
| DeepSeek | DeepSeek model integration |
| Hugging Face | Hosted inference API integration |
| Built-in Synthesizer | Offline fallback response generation |

### 🔍 2. Hybrid Vector Retrieval

Combines semantic and keyword-based retrieval to improve document search accuracy.

**Dense Retrieval – FAISS**
- Uses `sentence-transformers/all-MiniLM-L6-v2`.
- Converts document chunks into numerical embeddings.
- Retrieves semantically similar passages.
- Uses cosine similarity through normalized embeddings and inner-product search.

**Sparse Retrieval – BM25**
- Performs keyword-based document retrieval.
- Preserves exact terms, dates, names, codes, and numerical values.
- Helps retrieve information that semantic search might overlook.

**Reciprocal Rank Fusion (RRF)**
- Combines dense and sparse retrieval rankings.
- Improves retrieval coverage and relevance.

**Dynamic Alpha Weighting**

Users can adjust the balance between semantic and keyword-based retrieval:

- α = 0.0 → Pure sparse retrieval.
- α = 1.0 → Pure dense retrieval.
- 0 < α < 1 → Hybrid retrieval.

**Additional Retrieval Features**
- HyDE-based query expansion.
- Multi-query generation.
- Configurable retrieval parameters.

### 🎯 3. Deep NLI-Based Claim Verification

VerifAI uses a Natural Language Inference (NLI) cross-encoder to evaluate individual claims against retrieved evidence.

Model:

`cross-encoder/nli-MiniLM2-L6-H768`

Each claim is classified into one of three categories:

| Status | Color | Meaning |
|---|---|---|
| Supported | 🟢 Green | Evidence supports the claim |
| Unsupported / Neutral | 🟡 Yellow | Sufficient supporting evidence is unavailable |
| Contradicted | 🔴 Red | Evidence conflicts with the claim |

The system evaluates premise-hypothesis relationships to determine whether generated statements are supported by the retrieved passages.

### 🛡️ 4. Enterprise Guardrail Remediation

VerifAI includes three configurable verification policies.

| Policy | Description |
|---|---|
| Strict | Removes or rewrites claims without sufficient evidence |
| Balanced | Flags unsupported claims and corrects contradictions |
| Exploratory | Retains generated responses while displaying warnings |

**Automated Self-Correction**

When the hallucination risk exceeds the configured policy threshold:

1. Unsupported and contradicted claims are identified.
2. Retrieved evidence is used to generate a corrected response.
3. The revised answer is evaluated again.
4. The user receives a more evidence-grounded response.

### 💬 5. Conversational RAG Chatbot

- Supports multi-turn conversations.
- Maintains conversational history.
- Answers questions using uploaded knowledge sources.
- Displays groundedness indicators.
- Provides expandable source citations.
- Allows users to inspect retrieved evidence.

### 📚 6. Multi-Format Document Ingestion

Upload and process different types of knowledge sources.

**Supported formats:**
- PDF
- DOCX
- TXT
- CSV
- JSON
- Markdown
- Public Web URLs

**Knowledge Base Features:**
- Document catalog management.
- Automatic text extraction.
- Document chunking.
- Chunk inspection.
- Source-specific deletion.
- Vector index management.

**Built-in Benchmark Loaders:**
- James Webb Space Telescope (JWST) scientific overview.
- SQuAD validation dataset.

### 🧪 7. Automated Benchmarking and Evaluation

VerifAI includes a benchmarking module for evaluating the retrieval and verification pipeline.

Features include:

- SQuAD validation dataset evaluation.
- Automated question-answer testing.
- Faithfulness measurement.
- Hallucination risk measurement.
- Retrieval latency profiling.
- Generation latency profiling.
- Verification latency profiling.
- Interactive Plotly visualizations.

---

## 🏗️ System Architecture

VerifAI follows a modular Retrieval-Augmented Generation architecture.

```text
                 USER QUERY
                      |
                      v
             QUERY PROCESSING
                      |
                      v
             QUERY EXPANSION
                      |
             +--------+--------+
             |                 |
             v                 v
       DENSE RETRIEVAL    SPARSE RETRIEVAL
          (FAISS)             (BM25)
             |                 |
             +--------+--------+
                      |
                      v
              HYBRID RETRIEVAL
                      |
                      v
              RETRIEVED CONTEXT
                      |
                      v
                LLM GENERATION
                      |
                      v
               CLAIM EXTRACTION
                      |
                      v
             NLI CLAIM VERIFICATION
                      |
                      v
             HALLUCINATION ANALYSIS
                      |
                      v
               GUARDRAIL ENGINE
                      |
               +------+------+
               |             |
               v             v
          VERIFIED       CORRECTION
           ANSWER         REQUIRED
               |             |
               |        RE-GENERATION
               |             |
               +------+------+
                      |
                      v
              FINAL VERIFIED ANSWER
```

### 🖥️ Streamlit Application Modules

The application is organized into five main workspaces:

| Module | Functionality |
|---|---|
| ⚡ Deep Research & Verification | Query processing, hybrid retrieval, claim verification, evidence matrix, and latency analysis |
| 💬 Conversational RAG Chatbot | Multi-turn conversations with groundedness indicators and citations |
| 📚 Knowledge Base & Vector Explorer | Document uploads, web ingestion, benchmark loaders, and chunk inspection |
| 🧪 SQuAD Benchmark & Evaluation | Automated benchmarking, evaluation metrics, and visualizations |
| 🛡️ Guardrails & Policy Specs | Strict, Balanced, and Exploratory verification policies |

---

## 📊 Core Verification Metrics

### 1. Faithfulness (Groundedness Score)

Measures the percentage of generated claims that are supported by retrieved evidence.

\[
Faithfulness =
\frac{N_{supported}}{N_{total\ claims}}
\times 100
\]

A higher faithfulness score indicates that a larger proportion of the generated answer is supported by the available evidence.

### 2. Hallucination Risk Score

Measures the percentage of claims that are unsupported or contradicted by the retrieved evidence.

\[
Hallucination\ Risk =
\frac{N_{unsupported}+N_{contradicted}}
{N_{total\ claims}}
\times 100
\]

A lower hallucination risk score indicates better factual grounding.

### 3. Hybrid Retrieval Score

Balances semantic similarity and keyword matching using the alpha parameter.

\[
Score_{hybrid} =
\alpha \cdot Score_{dense}
+
(1-\alpha)\cdot Score_{sparse}
\]

Where:

- α represents the dense retrieval weight.
- (1 − α) represents the sparse retrieval weight.
- Dense score comes from FAISS.
- Sparse score comes from BM25.

---

## 📁 Repository Structure

```text
AI-RAG-Hallucination-Detection/
│
├── app.py
├── config.py
├── requirements.txt
├── test_pipeline.py
├── .env.example
├── train.csv
├── validation.csv
├── README.md
│
└── src/
    ├── __init__.py
    ├── document_loader.py
    ├── vector_store.py
    ├── llm_client.py
    ├── hallucination_checker.py
    └── rag_pipeline.py
```

### File Description

| File | Description |
|---|---|
| `app.py` | Main Streamlit application and user interface |
| `config.py` | Central configuration and provider settings |
| `requirements.txt` | Python dependencies |
| `test_pipeline.py` | Automated pipeline tests |
| `.env.example` | Environment variable template |
| `document_loader.py` | Document ingestion and text extraction |
| `vector_store.py` | FAISS and BM25 hybrid retrieval |
| `llm_client.py` | Multi-provider LLM integration |
| `hallucination_checker.py` | NLI-based claim verification |
| `rag_pipeline.py` | End-to-end RAG pipeline orchestration |

---

## 🚀 Installation & Quickstart

### Prerequisites

- Python 3.10 or higher.
- pip package manager.
- Git.
- Internet connection for downloading models and using cloud providers.

### 1. Clone the Repository

```bash
git clone https://github.com/kreepabaid/AI-RAG-Hallucination-Detection.git
```

Navigate into the project directory:

```bash
cd AI-RAG-Hallucination-Detection
```

### 2. Create a Virtual Environment

**Windows (PowerShell):**

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

**macOS / Linux:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

**Note:** PyTorch, Transformers, Sentence Transformers, and FAISS may require additional installation configuration depending on your operating system and hardware.

### 4. Configure Environment Variables

VerifAI includes a built-in offline synthesizer, allowing basic functionality without an external LLM API key.

For cloud-based inference, configure your preferred provider.

Create a `.env` file:

```bash
cp .env.example .env
```

On Windows PowerShell, you can also create the file manually:

```powershell
Copy-Item .env.example .env
```

Add your API keys:

```env
# Groq
GROQ_API_KEY=gsk_your_groq_key_here

# Google Gemini
GEMINI_API_KEY=your_gemini_key_here

# OpenAI / DeepSeek
OPENAI_API_KEY=your_api_key_here
OPENAI_BASE_URL=https://api.openai.com/v1

# Ollama
OLLAMA_BASE_URL=http://localhost:11434/v1

# Hugging Face
HF_TOKEN=hf_your_token_here
```

**Get API Keys:**

- Groq: https://console.groq.com/keys
- Google AI Studio: https://aistudio.google.com/app/apikey
- OpenAI: https://platform.openai.com/api-keys
- Hugging Face: https://huggingface.co/settings/tokens
- Ollama: https://ollama.com/

> Never upload your actual API keys or `.env` file to GitHub.

### 5. Run the Application

Start the Streamlit application:

```bash
streamlit run app.py
```

Open the application in your browser:

```text
http://localhost:8501
```

---

## 💡 How to Use VerifAI

Follow these steps to test the platform:

**Step 1 – Load Knowledge Sources**

Navigate to the Knowledge Base workspace.

- Upload a PDF, DOCX, TXT, CSV, or other supported document.
- Alternatively, load the built-in JWST benchmark data.
- Public web URLs can also be used as knowledge sources.

**Step 2 – Select an LLM Provider**

Choose an available provider:

- Groq
- Google Gemini
- Ollama
- OpenAI / DeepSeek
- Hugging Face
- Built-in Synthesizer

**Step 3 – Ask a Question**

Navigate to Deep Research & Verification.

Enter a question related to the uploaded documents.

Example:

```text
What rocket launched the James Webb Space Telescope and when?
```

Click the research and verification button.

**Step 4 – Inspect the Generated Answer**

Review:

- Generated response.
- Groundedness score.
- Hallucination risk score.
- Color-coded claim verification.
- Retrieved evidence passages.
- Claim-by-claim evidence matrix.

**Step 5 – Review Guardrail Corrections**

If unsupported or contradicted claims are detected, inspect the automatically corrected response according to the selected guardrail policy.

---

## 🧪 Running the Test Suite

Run the automated test suite:

```bash
python test_pipeline.py
```

The test suite is designed to validate:

1. Document loading and chunking.
2. Hybrid retrieval using FAISS and BM25.
3. NLI-based hallucination detection.
4. Multi-provider LLM integration.
5. End-to-end RAG pipeline execution.
6. Latency profiling.

Example test output (illustrative):

```text
=== STARTING VERIFAI RAG PIPELINE TESTS ===

--- 1. Document Loader & Chunking ---
SUCCESS: Document chunks created.

--- 2. Hybrid Vector Store ---
SUCCESS: Hybrid retrieval completed.

--- 3. Hallucination Checker ---
SUCCESS: Claim verification completed.

--- 4. Multi-Provider LLM Client ---
SUCCESS: Provider integration tested.

--- 5. End-to-End RAG Pipeline ---
SUCCESS: Pipeline execution completed.

=== TEST EXECUTION COMPLETED ===
```

Actual output may vary depending on the test data and execution environment.

---

## 🔬 Model Specifications

| Component | Model / Technology | Purpose |
|---|---|---|
| Dense Embeddings | `sentence-transformers/all-MiniLM-L6-v2` | Semantic document retrieval |
| Sparse Retrieval | BM25 | Keyword-based retrieval |
| NLI Cross-Encoder | `cross-encoder/nli-MiniLM2-L6-H768` | Claim-level verification |
| Vector Database | FAISS | Efficient similarity search |
| LLM Inference | Groq / Gemini / OpenAI / Ollama / HF | Response generation |
| Interface | Streamlit | Interactive web application |
| Visualization | Plotly | Evaluation charts and metrics |

---

## 🛠️ Technology Stack

**Programming Language**
- Python

**Frontend / Interface**
- Streamlit

**Machine Learning & NLP**
- PyTorch
- Transformers
- Sentence Transformers
- Natural Language Inference

**Information Retrieval**
- FAISS
- BM25
- Reciprocal Rank Fusion

**LLM Providers**
- Groq
- Google Gemini
- OpenAI
- DeepSeek
- Ollama
- Hugging Face

**Data Processing & Visualization**
- Pandas
- NumPy
- Plotly

---

## 🔮 Future Enhancements

Potential future improvements include:

- Advanced claim decomposition and verification.
- Improved evidence ranking and citation accuracy.
- Support for additional document formats.
- More advanced retrieval strategies.
- Explainable AI-based verification reports.
- Persistent knowledge base storage.
- Advanced benchmark datasets.
- Improved evaluation of retrieval and generation quality.

---

## 🤝 Contributing

Contributions, suggestions, bug reports, and feature requests are welcome.

To contribute:

1. Fork the repository.
2. Create a new feature branch.

```bash
git checkout -b feature/NewFeature
```

3. Commit your changes.

```bash
git commit -m "Add NewFeature"
```

4. Push your branch.

```bash
git push origin feature/NewFeature
```

5. Open a Pull Request.

---

## 📄 License

This project is distributed under the MIT License.

See the `LICENSE` file for more information.

---

## ⭐ Support

If you find VerifAI interesting or useful, consider giving this repository a ⭐ on GitHub.

**VerifAI – Building More Reliable, Transparent, and Trustworthy AI Systems.**

## 👩‍💻 Authors

This project was developed by:

- **Kreepa Baid**
- **Harshita Agarwal**
- **Sanskriti Sharma**
