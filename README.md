# 🔍 Hallucination-Aware RAG Pipeline & Web UI

An intelligent Retrieval-Augmented Generation (RAG) system with real-time **hallucination detection and automated correction**. The system extracts atomic factual claims from generated answers, verifies them using an NLI cross-encoder model against the retrieved source documents, flags ungrounded or contradicted statements, and produces an evidence-grounded answer.

---

## 🚀 Key Features

1. **Multi-Source Document Ingestion**:
   - **Files**: PDF (`.pdf`), Word (`.docx`), Plain Text (`.txt`, `.md`), CSV (`.csv`).
   - **Web URLs**: Direct web scraping with automatic tag cleanup.
   - **Raw Text**: Paste text directly into the web interface.
   - **One-Click Sample**: Built-in test article (JWST) for immediate demonstration.

2. **Vector Search & Embedding**:
   - Embeds document chunks using `sentence-transformers/all-MiniLM-L6-v2`.
   - Fast cosine-similarity vector retrieval via **FAISS** (`IndexFlatIP`).

3. **Hallucination Detection via Natural Language Inference (NLI)**:
   - Uses `cross-encoder/nli-MiniLM2-L6-H768` running locally on your CPU/GPU.
   - Parses the answer into atomic verifiable factual claims.
   - Verifies each claim against **all retrieved context chunks** without arbitrary truncation.
   - Classifies each claim into:
     - `SUPPORTED` (Entailment) with exact source evidence citations.
     - `UNSUPPORTED` (Neutral / Missing evidence in sources).
     - `CONTRADICTED` (Contradiction).

4. **Automated Hallucination Correction**:
   - Automatically generates a corrected, strictly source-grounded response when hallucinations are detected.
   - Visualizes a **Hallucination Risk Gauge** and claim-by-claim breakdown.

5. **Flexible Inference & Offline Fallback**:
   - Supports Hugging Face Inference API (`meta-llama/Llama-3.1-8B-Instruct`, `Qwen/Qwen2.5-7B-Instruct`, `mistralai/Mistral-7B-Instruct-v0.3`).
   - Includes an intelligent local extractive fallback mode so the app works seamlessly even without an API token!

---

## 📂 Project Architecture

```text
ai_project/
├── .venv/                      # Python virtual environment
├── src/                        # Core package
│   ├── __init__.py             # Package initializer and exports
│   ├── document_loader.py      # File parsers & chunking logic
│   ├── vector_store.py         # FAISS vector store & embeddings
│   ├── hallucination_checker.py # NLI cross-encoder fact checker
│   ├── llm_client.py           # Hugging Face inference & fallback LLM
│   └── rag_pipeline.py         # End-to-end RAG orchestrator
├── app.py                      # Interactive Streamlit Web UI
├── config.py                   # Global configuration & environment settings
├── test_pipeline.py            # Automated end-to-end test suite
├── requirements.txt            # Locked project dependencies
├── .env.example                # Template for environment variables
└── README.md                   # Project documentation
```

---

## 🛠️ Quick Start

### 1. Activate Environment
In PowerShell:
```powershell
.\.venv\Scripts\Activate.ps1
```

### 2. (Optional) Configure Hugging Face API Token
To use cloud LLMs (Llama-3.1, Qwen-2.5, etc.), you can get a free token from [Hugging Face Tokens](https://huggingface.co/settings/tokens):
- Either create a `.env` file:
  ```env
  HF_TOKEN=your_huggingface_token_here
  ```
- Or enter it directly in the Streamlit UI sidebar!

*(If no token is provided, the application automatically uses the local extractive fallback mode for full offline functionality).*

### 3. Run the Streamlit Web Application
```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 🧪 Running Automated Tests

Run the end-to-end verification suite:
```powershell
.\.venv\Scripts\python.exe test_pipeline.py
```

All 4 test phases (Loader, Vector Store, NLI Checker, Full Pipeline) will execute and output the validation results.
