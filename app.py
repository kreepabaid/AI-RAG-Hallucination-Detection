"""
app.py — Hallucination-Aware RAG Engine
Aesthetic, modern, bright SaaS AI research dashboard with real-time NLI claim verification.
"""

import os
import sys
import tempfile
from pathlib import Path
import streamlit as st

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
from src.rag_pipeline import HallucinationAwareRAG

# ── 1. Page Configuration (Wide Layout) ───────────────────────────────────────
st.set_page_config(
    page_title="Hallucination-Aware RAG — VerifAI",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── 2. Modern Aesthetic CSS System ────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

    /* Global reset & Canvas */
    .stApp {
        background-color: #F8FAFC !important;
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important;
        color: #0F172A !important;
    }

    /* Streamlit block container: wide, balanced padding */
    .block-container {
        max-width: 1440px !important;
        padding-top: 1.5rem !important;
        padding-bottom: 4rem !important;
        padding-left: 2.5rem !important;
        padding-right: 2.5rem !important;
    }

    /* Sidebar: crisp white, clean border, subtle shadow */
    [data-testid="stSidebar"] {
        background-color: #FFFFFF !important;
        border-right: 1px solid #E2E8F0 !important;
        box-shadow: 1px 0 3px rgba(0, 0, 0, 0.02) !important;
    }
    [data-testid="stSidebar"] > div:first-child {
        padding: 1.8rem 1.4rem !important;
    }

    /* Top Navigation Bar */
    .app-navbar {
        display: flex;
        align-items: center;
        justify-content: space-between;
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 14px 24px;
        margin-bottom: 24px;
        box-shadow: 0 1px 3px rgba(15, 23, 42, 0.04);
    }
    .nav-brand {
        display: flex;
        align-items: center;
        gap: 12px;
    }
    .nav-logo-box {
        width: 38px;
        height: 38px;
        border-radius: 10px;
        background: linear-gradient(135deg, #4F46E5 0%, #7C3AED 100%);
        display: flex;
        align-items: center;
        justify-content: center;
        color: #FFFFFF;
        font-size: 1.15rem;
        box-shadow: 0 4px 10px rgba(79, 70, 229, 0.25);
    }
    .nav-title {
        font-size: 1.15rem;
        font-weight: 700;
        color: #0F172A;
        letter-spacing: -0.02em;
        line-height: 1.2;
    }
    .nav-subtitle {
        font-size: 0.8rem;
        color: #64748B;
        font-weight: 500;
    }
    .nav-status-badge {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        background: #ECFDF5;
        color: #047857;
        font-size: 0.78rem;
        font-weight: 600;
        padding: 5px 12px;
        border-radius: 9999px;
        border: 1px solid #A7F3D0;
    }
    .pulse-dot {
        width: 7px;
        height: 7px;
        background-color: #10B981;
        border-radius: 50%;
        box-shadow: 0 0 0 2px rgba(16, 185, 129, 0.2);
    }

    /* Hero Header */
    .hero-box {
        margin-bottom: 28px;
    }
    .hero-tag {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: #EEF2FF;
        color: #4F46E5;
        font-size: 0.76rem;
        font-weight: 700;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        padding: 4px 10px;
        border-radius: 6px;
        border: 1px solid #C7D2FE;
        margin-bottom: 10px;
    }
    .hero-heading {
        font-size: 2.1rem;
        font-weight: 800;
        color: #0F172A;
        letter-spacing: -0.03em;
        line-height: 1.2;
        margin: 0 0 6px 0;
    }
    .hero-desc {
        font-size: 1.02rem;
        color: #475569;
        max-width: 860px;
        line-height: 1.6;
        margin: 0;
    }

    /* Cards */
    .ui-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 14px;
        padding: 24px;
        box-shadow: 0 1px 3px rgba(15, 23, 42, 0.03);
        margin-bottom: 24px;
    }
    .ui-card-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 16px;
        padding-bottom: 12px;
        border-bottom: 1px solid #F1F5F9;
    }
    .ui-card-title {
        font-size: 1.05rem;
        font-weight: 700;
        color: #0F172A;
        letter-spacing: -0.01em;
    }
    .ui-card-subtitle {
        font-size: 0.85rem;
        color: #64748B;
        margin-top: 2px;
    }

    /* Metric Cards Grid */
    .metric-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 16px;
        margin-bottom: 24px;
    }
    .metric-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 18px 20px;
        box-shadow: 0 1px 3px rgba(15, 23, 42, 0.02);
        position: relative;
        overflow: hidden;
    }
    .metric-card-stripe {
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
    }
    .metric-card-label {
        font-size: 0.74rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #64748B;
        margin-bottom: 6px;
    }
    .metric-card-value {
        font-size: 1.85rem;
        font-weight: 800;
        color: #0F172A;
        line-height: 1.1;
    }
    .metric-card-meta {
        font-size: 0.78rem;
        color: #94A3B8;
        margin-top: 5px;
        font-weight: 500;
    }

    /* Modern Risk Precision Bar Card */
    .precision-risk-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 20px 24px;
        margin-bottom: 24px;
        box-shadow: 0 1px 3px rgba(15, 23, 42, 0.03);
    }
    .risk-bar-track {
        height: 8px;
        background-color: #F1F5F9;
        border-radius: 9999px;
        overflow: hidden;
        margin: 10px 0 6px 0;
    }
    .risk-bar-fill {
        height: 100%;
        border-radius: 9999px;
        transition: width 0.4s ease-in-out;
    }

    /* Answers Dual View */
    .answer-panel {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 22px;
        height: 100%;
        box-shadow: 0 1px 3px rgba(15, 23, 42, 0.02);
    }
    .answer-panel-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 14px;
        padding-bottom: 10px;
        border-bottom: 1px solid #F1F5F9;
    }
    .badge-raw {
        background: #F1F5F9;
        color: #475569;
        font-size: 0.72rem;
        font-weight: 700;
        padding: 3px 9px;
        border-radius: 6px;
        letter-spacing: 0.03em;
    }
    .badge-grounded {
        background: #ECFDF5;
        color: #047857;
        font-size: 0.72rem;
        font-weight: 700;
        padding: 3px 9px;
        border-radius: 6px;
        border: 1px solid #A7F3D0;
        letter-spacing: 0.03em;
    }
    .answer-text-content {
        font-size: 0.94rem;
        line-height: 1.7;
        color: #1E293B;
        white-space: pre-wrap;
    }

    /* Citation Pills */
    .cite-badge {
        display: inline-flex;
        align-items: center;
        gap: 5px;
        background: #F8FAFC;
        color: #334155;
        font-size: 0.76rem;
        font-weight: 600;
        padding: 4px 10px;
        border-radius: 6px;
        border: 1px solid #E2E8F0;
        margin-right: 6px;
        margin-bottom: 6px;
    }
    .cite-evidence-box {
        background: #F8FAFC;
        border-left: 3px solid #6366F1;
        border-radius: 6px;
        padding: 12px 16px;
        font-size: 0.88rem;
        color: #1E293B;
        margin-top: 10px;
        line-height: 1.6;
    }

    /* Confidence Bars */
    .confidence-row {
        display: flex;
        align-items: center;
        gap: 12px;
        font-size: 0.82rem;
        margin-bottom: 6px;
    }
    .confidence-label {
        width: 110px;
        color: #64748B;
        font-weight: 500;
    }
    .confidence-track {
        flex-grow: 1;
        height: 6px;
        background-color: #F1F5F9;
        border-radius: 9999px;
        overflow: hidden;
    }
    .confidence-val {
        width: 48px;
        text-align: right;
        font-weight: 600;
        color: #0F172A;
    }

    /* Source item rows */
    .source-row {
        display: flex;
        align-items: center;
        justify-content: space-between;
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 10px 14px;
        margin-bottom: 8px;
    }
    .source-row-left {
        display: flex;
        align-items: center;
        gap: 10px;
    }
    .source-row-name {
        font-size: 0.88rem;
        font-weight: 600;
        color: #0F172A;
    }
    .source-row-meta {
        font-size: 0.76rem;
        color: #64748B;
    }

    /* Streamlit component stylings */
    /* Tabs */
    div[data-baseweb="tab-list"] {
        gap: 20px !important;
        background-color: transparent !important;
        border-bottom: 1px solid #E2E8F0 !important;
    }
    div[data-baseweb="tab"] {
        font-size: 0.9rem !important;
        font-weight: 600 !important;
        color: #64748B !important;
        padding: 8px 4px 12px 4px !important;
    }
    div[data-baseweb="tab"][aria-selected="true"] {
        color: #4F46E5 !important;
        border-bottom-color: #4F46E5 !important;
    }

    /* Buttons */
    button[kind="primary"] {
        background: linear-gradient(135deg, #4F46E5 0%, #4338CA 100%) !important;
        color: #FFFFFF !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 8px 18px !important;
        font-weight: 600 !important;
        font-size: 0.88rem !important;
        box-shadow: 0 1px 3px rgba(79, 70, 229, 0.25) !important;
        transition: all 0.15s ease-in-out !important;
    }
    button[kind="primary"]:hover {
        box-shadow: 0 4px 12px rgba(79, 70, 229, 0.35) !important;
        transform: translateY(-1px) !important;
    }
    button[kind="secondary"] {
        background: #FFFFFF !important;
        color: #334155 !important;
        border: 1px solid #CBD5E1 !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        font-size: 0.85rem !important;
        transition: all 0.15s ease-in-out !important;
    }
    button[kind="secondary"]:hover {
        background: #F8FAFC !important;
        border-color: #94A3B8 !important;
        color: #0F172A !important;
    }

    /* Expanders */
    [data-testid="stExpander"] {
        background: #FFFFFF !important;
        border: 1px solid #E2E8F0 !important;
        border-radius: 10px !important;
        margin-bottom: 10px !important;
        box-shadow: 0 1px 2px rgba(0,0,0,0.02) !important;
    }
    [data-testid="stExpander"] summary {
        font-weight: 600 !important;
        color: #1E293B !important;
        padding: 10px 14px !important;
    }

    /* Sidebar info badges */
    .sidebar-spec-box {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 10px 12px;
        margin-bottom: 8px;
    }
    .sidebar-spec-label {
        font-size: 0.72rem;
        font-weight: 700;
        text-transform: uppercase;
        color: #64748B;
        letter-spacing: 0.04em;
    }
    .sidebar-spec-val {
        font-size: 0.82rem;
        font-weight: 600;
        color: #0F172A;
        margin-top: 2px;
        word-break: break-all;
    }
</style>
""", unsafe_allow_html=True)

# ── 3. Benchmark Sample Data ──────────────────────────────────────────────────
SAMPLE_DOCUMENT = """The James Webb Space Telescope (JWST) is an advanced optical and infrared space observatory developed primarily to conduct deep-field infrared astronomy and analyze exoplanet atmospheres.
As the most capable optical observatory deployed in space, its high resolution and infrared sensitivity allow it to view astronomical objects too old, distant, or faint for the Hubble Space Telescope.
The observatory was launched on 25 December 2021 aboard an Ariane 5 rocket from the Guiana Space Centre in Kourou, French Guiana, and arrived at the Sun-Earth L2 Lagrange point in January 2022.
JWST's primary mirror comprises 18 hexagonal segments made of gold-plated beryllium, creating an effective aperture diameter of 6.5 meters (21 feet).
The telescope operates in a halo orbit around the Sun-Earth L2 point, approximately 1.5 million kilometers (930,000 miles) beyond Earth's orbit, using a three-mirror anastigmat optical system."""

SAMPLE_QUESTION = "When was the James Webb Space Telescope launched and what is its mirror made of?"

# ── 4. Pipeline Initialization ────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading Vector Store & NLI Verification Engine...")
def get_rag_pipeline():
    return HallucinationAwareRAG()

rag = get_rag_pipeline()

# ── 5. Session State Management ───────────────────────────────────────────────
if "sources" not in st.session_state:
    st.session_state.sources = []
if "source_details" not in st.session_state:
    st.session_state.source_details = []
if "result" not in st.session_state:
    st.session_state.result = None
if "api_key" not in st.session_state:
    st.session_state.api_key = config.HF_API_KEY
if "selected_model" not in st.session_state:
    st.session_state.selected_model = config.LLM_MODEL
if "question_input" not in st.session_state:
    st.session_state.question_input = ""

# ── 6. Top Navigation Bar ─────────────────────────────────────────────────────
st.markdown("""
<div class="app-navbar">
    <div class="nav-brand">
        <div class="nav-logo-box">⚡</div>
        <div>
            <div class="nav-title">VerifAI · Hallucination-Aware RAG</div>
            <div class="nav-subtitle">Retrieval Grounding & NLI Cross-Encoder Verification</div>
        </div>
    </div>
    <div>
        <span class="nav-status-badge">
            <span class="pulse-dot"></span>
            System Operational
        </span>
    </div>
</div>
""", unsafe_allow_html=True)

# ── 7. Sidebar Configuration ──────────────────────────────────────────────────
with st.sidebar:
    st.markdown("<div style='font-size: 0.75rem; font-weight: 700; color: #475569; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 12px;'>Model & API Setup</div>", unsafe_allow_html=True)

    # API Key management
    api_key_input = st.text_input(
        "Hugging Face API Token",
        value=st.session_state.api_key,
        type="password",
        help="Free User Access Token from https://huggingface.co/settings/tokens",
    )
    if api_key_input != st.session_state.api_key:
        st.session_state.api_key = api_key_input
        rag.set_api_key(api_key_input)

    # Model selection
    selected_model = st.selectbox(
        "LLM Generator",
        options=config.AVAILABLE_MODELS,
        index=config.AVAILABLE_MODELS.index(st.session_state.selected_model)
        if st.session_state.selected_model in config.AVAILABLE_MODELS else 0,
    )
    if selected_model != st.session_state.selected_model:
        st.session_state.selected_model = selected_model
        rag.set_model(selected_model)

    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

    # Retrieval slider
    top_k = st.slider("Retrieval Depth (Top-K Chunks)", min_value=1, max_value=10, value=config.TOP_K)

    # Test API button
    if st.button("Test API Connection", use_container_width=True, type="primary"):
        with st.spinner("Pinging Hugging Face endpoint..."):
            valid, msg = rag.llm.validate_connection()
            if valid:
                st.success(f"Connected: {msg}")
            else:
                st.warning(f"Notice: {msg}")

    with st.expander("Token Guide & Fallback Info"):
        st.markdown("""
        - Obtain a free token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).
        - If no token is provided, the engine runs in **local extractive fallback mode** for offline testing.
        """)

    st.markdown("<hr style='border: none; border-top: 1px solid #E2E8F0; margin: 18px 0;'>", unsafe_allow_html=True)
    st.markdown("<div style='font-size: 0.75rem; font-weight: 700; color: #475569; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 10px;'>System Architecture</div>", unsafe_allow_html=True)

    st.markdown(f"""
    <div class="sidebar-spec-box">
        <div class="sidebar-spec-label">Embeddings</div>
        <div class="sidebar-spec-val">{config.EMBEDDING_MODEL.split('/')[-1]}</div>
    </div>
    <div class="sidebar-spec-box">
        <div class="sidebar-spec-label">NLI Verifier</div>
        <div class="sidebar-spec-val">{config.NLI_MODEL.split('/')[-1]}</div>
    </div>
    <div class="sidebar-spec-box">
        <div class="sidebar-spec-label">Indexed Vector Chunks</div>
        <div class="sidebar-spec-val">{rag.vector_store.total_chunks} chunks in memory</div>
    </div>
    <div class="sidebar-spec-box">
        <div class="sidebar-spec-label">Active Sources</div>
        <div class="sidebar-spec-val">{len(st.session_state.sources)} document(s)</div>
    </div>
    """, unsafe_allow_html=True)

# ── 8. Hero Section ───────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-box">
    <span class="hero-tag">Evidence Grounding · Natural Language Inference</span>
    <h1 class="hero-heading">Grounding & Verification Engine</h1>
    <p class="hero-desc">Ingest trusted sources, query the knowledge base, and verify generated claims statement-by-statement with cross-encoder NLI.</p>
</div>
""", unsafe_allow_html=True)

# ── 9. Knowledge Ingestion Card ───────────────────────────────────────────────
st.markdown("""
<div class="ui-card">
    <div class="ui-card-header">
        <div>
            <div class="ui-card-title">Knowledge Repository</div>
            <div class="ui-card-subtitle">Index documents and websites into the local FAISS vector space.</div>
        </div>
    </div>
""", unsafe_allow_html=True)

tab_files, tab_url, tab_text, tab_sample = st.tabs(["Upload Files", "Web Scraping", "Direct Text", "Benchmark Dataset"])

# Tab 1: Upload Files
with tab_files:
    uploaded_files = st.file_uploader(
        "Upload files (PDF, DOCX, TXT, CSV, MD)",
        type=["pdf", "docx", "txt", "csv", "md"],
        accept_multiple_files=True,
        label_visibility="collapsed",
        key="uploader_input"
    )
    if st.button("Index Files", type="primary", key="btn_idx_files"):
        if not uploaded_files:
            st.warning("Please select at least one file to index.")
        else:
            for f in uploaded_files:
                ext = "." + f.name.rsplit(".", 1)[-1]
                with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
                    tmp.write(f.read())
                    tmp_path = tmp.name
                with st.spinner(f"Indexing {f.name}..."):
                    try:
                        n = rag.add_source(tmp_path, original_name=f.name)
                        if f.name not in st.session_state.sources:
                            st.session_state.sources.append(f.name)
                            kb_size = round(f.size / 1024, 1) if hasattr(f, 'size') else 0.0
                            st.session_state.source_details.append({
                                "name": f.name,
                                "meta": f"{kb_size} KB · {n} chunks",
                                "chunks": n
                            })
                        st.success(f"Indexed {f.name} ({n} chunks)")
                    except Exception as e:
                        st.error(f"Error parsing {f.name}: {e}")
                    finally:
                        try:
                            os.unlink(tmp_path)
                        except OSError:
                            pass

# Tab 2: Web Scraping
with tab_url:
    url_input = st.text_input("Public Web URL", placeholder="https://en.wikipedia.org/wiki/Artificial_intelligence")
    if st.button("Scrape & Index URL", type="primary", key="btn_idx_url"):
        if not url_input.strip():
            st.warning("Please provide a valid web URL.")
        else:
            with st.spinner(f"Scraping & indexing {url_input}..."):
                try:
                    n = rag.add_source(url_input.strip())
                    clean_url = url_input.strip()
                    if clean_url not in st.session_state.sources:
                        st.session_state.sources.append(clean_url)
                        st.session_state.source_details.append({
                            "name": clean_url,
                            "meta": f"Web Page · {n} chunks",
                            "chunks": n
                        })
                    st.success(f"Indexed web source ({n} chunks)")
                except Exception as e:
                    st.error(f"Failed to scrape URL: {e}")

# Tab 3: Direct Text
with tab_text:
    txt_label = st.text_input("Document Name", value="Custom Reference Note")
    txt_body = st.text_area("Content Excerpt", height=110, placeholder="Paste reference documentation or excerpts here...")
    if st.button("Index Text Snippet", type="primary", key="btn_idx_text"):
        if not txt_body.strip():
            st.warning("Please enter text content before indexing.")
        else:
            with st.spinner("Embedding text into vector store..."):
                lbl = txt_label.strip() or "Custom Reference"
                n = rag.add_raw_text(txt_body.strip(), source_name=lbl)
                if lbl not in st.session_state.sources:
                    st.session_state.sources.append(lbl)
                    st.session_state.source_details.append({
                        "name": lbl,
                        "meta": f"{len(txt_body.split())} words · {n} chunks",
                        "chunks": n
                    })
                st.success(f"Indexed '{lbl}' ({n} chunks)")

# Tab 4: Benchmark Sample
with tab_sample:
    st.markdown("Load our built-in benchmark dataset about the **James Webb Space Telescope (JWST)** for immediate verification testing.")
    if st.button("Load JWST Benchmark Document", key="btn_idx_sample"):
        with st.spinner("Indexing benchmark document..."):
            n = rag.add_raw_text(SAMPLE_DOCUMENT, source_name="JWST_Overview.txt")
            if "JWST_Overview.txt" not in st.session_state.sources:
                st.session_state.sources.append("JWST_Overview.txt")
                st.session_state.source_details.append({
                    "name": "JWST_Overview.txt",
                    "meta": f"Benchmark · {n} chunks",
                    "chunks": n
                })
            st.session_state.question_input = SAMPLE_QUESTION
            st.success(f"Loaded JWST benchmark ({n} chunks). Ready for questioning below.")

# Active Sources List
if st.session_state.sources:
    st.markdown("<div style='font-size: 0.84rem; font-weight: 700; color: #475569; text-transform: uppercase; letter-spacing: 0.04em; margin-top: 20px; margin-bottom: 8px;'>Active Knowledge Sources</div>", unsafe_allow_html=True)
    for src in st.session_state.source_details:
        st.markdown(f"""
        <div class="source-row">
            <div class="source-row-left">
                <span style="color: #4F46E5; font-size: 1.1rem;">📄</span>
                <div>
                    <div class="source-row-name">{src['name']}</div>
                    <div class="source-row-meta">{src['meta']}</div>
                </div>
            </div>
            <span style="font-size: 0.74rem; font-weight: 600; background: #ECFDF5; color: #047857; padding: 3px 8px; border-radius: 6px; border: 1px solid #A7F3D0;">Indexed</span>
        </div>
        """, unsafe_allow_html=True)

    col_btn_clr, _ = st.columns([1, 4])
    with col_btn_clr:
        if st.button("Clear All Sources", key="btn_wipe_sources"):
            rag.clear_sources()
            st.session_state.sources = []
            st.session_state.source_details = []
            st.session_state.result = None
            st.rerun()

st.markdown("</div>", unsafe_allow_html=True)

# ── 10. Query & Analysis Card ─────────────────────────────────────────────────
st.markdown("""
<div class="ui-card">
    <div class="ui-card-header">
        <div>
            <div class="ui-card-title">Query & Verification</div>
            <div class="ui-card-subtitle">Query the knowledge base and verify generated statements against evidence chunks.</div>
        </div>
    </div>
""", unsafe_allow_html=True)

if not st.session_state.sources:
    st.markdown("""
    <div style="background: #F8FAFC; border: 1px dashed #CBD5E1; border-radius: 10px; padding: 24px; text-align: center;">
        <div style="font-size: 0.95rem; font-weight: 600; color: #334155; margin-bottom: 2px;">No documents indexed yet</div>
        <div style="font-size: 0.82rem; color: #64748B;">Upload a document or click 'Load JWST Benchmark Document' above to begin.</div>
    </div>
    """, unsafe_allow_html=True)
else:
    q_col, btn_col = st.columns([5, 1.4])
    with q_col:
        question_text = st.text_input(
            "Query Input",
            value=st.session_state.question_input,
            placeholder="Ask a question grounded in your indexed sources...",
            label_visibility="collapsed",
            key="query_box_input"
        )
    with btn_col:
        run_analysis = st.button("Analyze & Verify", type="primary", use_container_width=True, key="btn_run_rag")

    if run_analysis:
        if not question_text.strip():
            st.warning("Please type a question before running analysis.")
        else:
            with st.spinner("Retrieving evidence → Generating response → Verifying factual claims..."):
                try:
                    res_dict = rag.query(question_text.strip(), top_k=top_k)
                    st.session_state.result = res_dict
                except Exception as ex:
                    st.error(f"Analysis error: {ex}")

st.markdown("</div>", unsafe_allow_html=True)

# ── 11. Results & Verification Dashboard ──────────────────────────────────────
res = st.session_state.result
if res:
    if "error" in res:
        st.error(res["error"])
    else:
        claims = res.get("claims", [])
        total_claims = len(claims)
        supported_count = sum(1 for c in claims if c.get("verdict") == "supported")
        issues_count = sum(1 for c in claims if c.get("verdict") != "supported")
        contra_count = sum(1 for c in claims if c.get("verdict") == "contradicted")
        neutral_count = sum(1 for c in claims if c.get("verdict") == "unsupported")
        risk_score = res.get("hallucination_score", 0.0)

        # Semantics
        if risk_score <= 25.0:
            status_text = "High Fidelity"
            status_color = "#10B981"
            status_bg = "#ECFDF5"
            status_border = "#A7F3D0"
        elif risk_score <= 50.0:
            status_text = "Moderate Risk"
            status_color = "#F59E0B"
            status_bg = "#FFFBEB"
            status_border = "#FDE68A"
        else:
            status_text = "High Hallucination Risk"
            status_color = "#F43F5E"
            status_bg = "#FFF1F2"
            status_border = "#FECDD3"

        # 4-Column Metric Grid
        st.markdown(f"""
        <div class="metric-grid">
            <div class="metric-card">
                <div class="metric-card-stripe" style="background-color: #6366F1;"></div>
                <div class="metric-card-label">Claims Extracted</div>
                <div class="metric-card-value">{total_claims}</div>
                <div class="metric-card-meta">Atomic factual claims</div>
            </div>
            <div class="metric-card">
                <div class="metric-card-stripe" style="background-color: #10B981;"></div>
                <div class="metric-card-label">Grounded / Supported</div>
                <div class="metric-card-value" style="color: #10B981;">{supported_count}</div>
                <div class="metric-card-meta">Fully backed by sources</div>
            </div>
            <div class="metric-card">
                <div class="metric-card-stripe" style="background-color: #F59E0B;"></div>
                <div class="metric-card-label">Uncertain / Neutral</div>
                <div class="metric-card-value" style="color: #F59E0B;">{neutral_count}</div>
                <div class="metric-card-meta">Missing source proof</div>
            </div>
            <div class="metric-card">
                <div class="metric-card-stripe" style="background-color: {status_color};"></div>
                <div class="metric-card-label">Hallucination Risk</div>
                <div class="metric-card-value" style="color: {status_color};">{risk_score}%</div>
                <div class="metric-card-meta">{status_text}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Precision Risk Bar Card
        st.markdown(f"""
        <div class="precision-risk-card">
            <div style="display: flex; align-items: center; justify-content: space-between;">
                <div>
                    <div style="font-size: 0.74rem; font-weight: 700; color: #64748B; text-transform: uppercase; letter-spacing: 0.05em;">Hallucination Risk Score</div>
                    <div style="font-size: 1.6rem; font-weight: 800; color: {status_color}; margin-top: 2px;">{risk_score}%</div>
                </div>
                <span style="font-size: 0.78rem; font-weight: 700; color: {status_color}; background: {status_bg}; border: 1px solid {status_border}; padding: 4px 12px; border-radius: 9999px;">
                    {status_text}
                </span>
            </div>
            <div class="risk-bar-track">
                <div class="risk-bar-fill" style="width: {max(3.0, risk_score)}%; background-color: {status_color};"></div>
            </div>
            <div style="display: flex; justify-content: space-between; font-size: 0.75rem; color: #94A3B8; font-weight: 500;">
                <span>0% (Fully Grounded)</span>
                <span>50% (Moderate Risk)</span>
                <span>100% (Unverified)</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        if res.get("is_fallback"):
            st.info("Operating in local extractive fallback mode. Connect your free Hugging Face token in the sidebar for full conversational generation.")

        # Side-by-Side Answers
        col_orig, col_fixed = st.columns(2)

        with col_orig:
            st.markdown(f"""
            <div class="answer-panel">
                <div class="answer-panel-header">
                    <div style="font-size: 0.95rem; font-weight: 700; color: #0F172A;">Raw LLM Generation</div>
                    <span class="badge-raw">Initial Output</span>
                </div>
                <div class="answer-text-content">{res.get('answer', '')}</div>
            </div>
            """, unsafe_allow_html=True)

        with col_fixed:
            corrected = res.get("corrected_answer") or res.get("answer", "")
            st.markdown(f"""
            <div class="answer-panel" style="border-color: #A7F3D0; background: #FAFDFB;">
                <div class="answer-panel-header">
                    <div style="font-size: 0.95rem; font-weight: 700; color: #065F46;">Grounded & Corrected Output</div>
                    <span class="badge-grounded">Verified Grounding</span>
                </div>
                <div class="answer-text-content">{corrected}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<div style='height: 24px;'></div>", unsafe_allow_html=True)

        # Claim-Level Verification Matrix
        st.markdown("""
        <div style="margin-bottom: 14px;">
            <div style="font-size: 1.15rem; font-weight: 700; color: #0F172A;">Claim-by-Claim Verification Matrix</div>
            <div style="font-size: 0.84rem; color: #64748B; margin-top: 2px;">Each extracted claim verified across all retrieved vector passages using cross-encoder NLI.</div>
        </div>
        """, unsafe_allow_html=True)

        for i, c in enumerate(claims, 1):
            verdict = c.get("verdict", "unsupported")
            conf = c.get("confidence", 0.0)
            scores = c.get("scores", {"supported": 0.0, "unsupported": 0.0, "contradicted": 0.0})

            if verdict == "supported":
                verdict_badge = "<span style='background:#ECFDF5; color:#047857; font-size:0.72rem; font-weight:700; padding:3px 8px; border-radius:4px; border:1px solid #A7F3D0;'>SUPPORTED</span>"
                header_icon = "✓"
            elif verdict == "contradicted":
                verdict_badge = "<span style='background:#FFF1F2; color:#9F1239; font-size:0.72rem; font-weight:700; padding:3px 8px; border-radius:4px; border:1px solid #FECDD3;'>CONTRADICTED</span>"
                header_icon = "✕"
            else:
                verdict_badge = "<span style='background:#FFFBEB; color:#92400E; font-size:0.72rem; font-weight:700; padding:3px 8px; border-radius:4px; border:1px solid #FDE68A;'>NEUTRAL / UNVERIFIED</span>"
                header_icon = "⚠"

            claim_text = c.get("claim", "")
            with st.expander(f"{header_icon} Claim {i}: {claim_text[:85]}{'...' if len(claim_text) > 85 else ''}"):
                st.markdown(f"**Extracted Claim:** {claim_text}")
                st.markdown(f"**Verification Outcome:** {verdict_badge} · **Confidence:** `{conf}%`", unsafe_allow_html=True)

                # Confidence Distribution Bars
                s_pct = scores.get("supported", 0.0)
                n_pct = scores.get("unsupported", 0.0)
                c_pct = scores.get("contradicted", 0.0)

                st.markdown(f"""
                <div style="margin: 12px 0 10px 0;">
                    <div class="confidence-row">
                        <span class="confidence-label">Supported</span>
                        <div class="confidence-track"><div style="width:{s_pct}%; height:100%; background-color:#10B981; border-radius:9999px;"></div></div>
                        <span class="confidence-val">{s_pct}%</span>
                    </div>
                    <div class="confidence-row">
                        <span class="confidence-label">Neutral</span>
                        <div class="confidence-track"><div style="width:{n_pct}%; height:100%; background-color:#F59E0B; border-radius:9999px;"></div></div>
                        <span class="confidence-val">{n_pct}%</span>
                    </div>
                    <div class="confidence-row">
                        <span class="confidence-label">Contradicted</span>
                        <div class="confidence-track"><div style="width:{c_pct}%; height:100%; background-color:#F43F5E; border-radius:9999px;"></div></div>
                        <span class="confidence-val">{c_pct}%</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # Cited Evidence
                src = c.get("evidence_source", "Knowledge Base")
                passage = c.get("evidence_passage", "No corroborating passage identified.")
                st.markdown(f"""
                <div style="font-size: 0.78rem; font-weight: 700; color: #475569; text-transform: uppercase; margin-top: 10px;">
                    Cited Passage Evidence <span class="cite-badge" style="margin-left: 6px;">Source: {src}</span>
                </div>
                <div class="cite-evidence-box">"{passage}"</div>
                """, unsafe_allow_html=True)

        st.markdown("<div style='height: 18px;'></div>", unsafe_allow_html=True)

        # Retrieved Evidence Inspection
        with st.expander("Retrieved Vector Context Chunks", expanded=False):
            evidence_items = res.get("retrieved_evidence", [])
            if evidence_items:
                for ev in evidence_items:
                    cid = ev.get("chunk_id", 1)
                    src_name = ev.get("source", "Document")
                    sim_pct = ev.get("similarity_pct", 0.0)
                    sim_score = ev.get("similarity", 0.0)

                    st.markdown(f"""
                    <div style="background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 8px; padding: 12px 16px; margin-bottom: 10px;">
                        <div style="display: flex; align-items: center; gap: 6px; margin-bottom: 6px; flex-wrap: wrap;">
                            <span class="cite-badge">Source: {src_name}</span>
                            <span class="cite-badge">Chunk #{cid:02d}</span>
                            <span class="cite-badge" style="background:#ECFDF5; color:#047857; border-color:#A7F3D0;">Cosine Score: {sim_score:.4f} ({sim_pct}%)</span>
                        </div>
                        <div style="font-size: 0.88rem; line-height: 1.6; color: #334155; background: #F8FAFC; padding: 10px 12px; border-radius: 6px;">
                            {ev.get('text', '')}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.text(res.get("context_used", "No retrieved passages available."))