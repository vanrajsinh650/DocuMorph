import streamlit as st
import json
import tempfile
import os
import time
from pathlib import Path
from io import BytesIO

# Import core functions from main.py
from main import (
    GroqKeyPool,
    extract_text_groq,
    extract_text_tesseract,
    parse_questions,
    validate_questions,
    save_raw_text,
    POPPLER_PATH,
)

# ─── PAGE CONFIG ─────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="DocuMorph",
    page_icon="📄",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ─── CUSTOM CSS ──────────────────────────────────────────────────────────────

st.markdown("""
<style>
    /* Reset streamlit defaults */
    .stApp {
        background-color: #0a0a0a;
        color: #e0e0e0;
    }
    
    /* Hide streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Main container */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 720px;
    }
    
    /* Typography */
    h1, h2, h3, h4, h5, h6 {
        color: #ffffff !important;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        font-weight: 600;
    }
    
    p, span, label, .stMarkdown {
        color: #b0b0b0 !important;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* App title */
    .app-title {
        font-size: 1.8rem;
        font-weight: 700;
        color: #ffffff;
        margin-bottom: 0.2rem;
        letter-spacing: -0.02em;
    }
    
    .app-subtitle {
        font-size: 0.9rem;
        color: #666666;
        margin-bottom: 2rem;
    }
    
    /* Divider */
    .divider {
        border: none;
        border-top: 1px solid #1a1a1a;
        margin: 1.5rem 0;
    }
    
    /* File uploader */
    .stFileUploader > div {
        background-color: #111111 !important;
        border: 1px solid #222222 !important;
        border-radius: 8px !important;
    }
    
    .stFileUploader label {
        color: #999999 !important;
    }
    
    /* Buttons */
    .stButton > button {
        background-color: #ffffff !important;
        color: #000000 !important;
        border: none !important;
        border-radius: 6px !important;
        font-weight: 600 !important;
        font-size: 0.85rem !important;
        padding: 0.5rem 1.5rem !important;
        width: 100%;
        transition: opacity 0.2s;
    }
    
    .stButton > button:hover {
        opacity: 0.85;
    }
    
    .stDownloadButton > button {
        background-color: #ffffff !important;
        color: #000000 !important;
        border: none !important;
        border-radius: 6px !important;
        font-weight: 600 !important;
        font-size: 0.85rem !important;
        padding: 0.5rem 1.5rem !important;
        width: 100%;
    }
    
    /* Select box */
    .stSelectbox > div > div {
        background-color: #111111 !important;
        border-color: #222222 !important;
        color: #e0e0e0 !important;
    }
    
    /* Number input */
    .stNumberInput > div > div > input {
        background-color: #111111 !important;
        border-color: #222222 !important;
        color: #e0e0e0 !important;
    }
    
    /* Text input */
    .stTextInput > div > div > input {
        background-color: #111111 !important;
        border-color: #222222 !important;
        color: #e0e0e0 !important;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background-color: #111111 !important;
        border-color: #222222 !important;
        color: #e0e0e0 !important;
    }
    
    /* Status boxes */
    .status-box {
        background-color: #111111;
        border: 1px solid #1a1a1a;
        border-radius: 8px;
        padding: 1rem 1.2rem;
        margin: 0.5rem 0;
    }
    
    .status-label {
        font-size: 0.75rem;
        color: #555555;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.3rem;
    }
    
    .status-value {
        font-size: 1.1rem;
        color: #ffffff;
        font-weight: 600;
    }
    
    /* Stats grid */
    .stat-card {
        background-color: #111111;
        border: 1px solid #1a1a1a;
        border-radius: 8px;
        padding: 0.8rem 1rem;
        text-align: center;
    }
    
    .stat-number {
        font-size: 1.5rem;
        font-weight: 700;
        color: #ffffff;
    }
    
    .stat-label {
        font-size: 0.7rem;
        color: #555555;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    /* Progress */
    .stProgress > div > div {
        background-color: #222222 !important;
    }
    
    .stProgress > div > div > div {
        background-color: #ffffff !important;
    }
    
    /* Alert/info boxes */
    .stAlert {
        background-color: #111111 !important;
        border-color: #222222 !important;
        color: #b0b0b0 !important;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        background-color: transparent;
        gap: 0;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: transparent !important;
        color: #666666 !important;
        border-bottom: 2px solid transparent;
        padding: 0.5rem 1rem;
    }
    
    .stTabs [aria-selected="true"] {
        color: #ffffff !important;
        border-bottom-color: #ffffff !important;
    }
    
    /* JSON preview */
    .json-preview {
        background-color: #0d0d0d;
        border: 1px solid #1a1a1a;
        border-radius: 8px;
        padding: 1rem;
        font-family: 'JetBrains Mono', 'Fira Code', monospace;
        font-size: 0.8rem;
        color: #b0b0b0;
        max-height: 400px;
        overflow-y: auto;
        white-space: pre-wrap;
        word-break: break-word;
    }
    
    /* Question preview card */
    .q-card {
        background-color: #111111;
        border: 1px solid #1a1a1a;
        border-radius: 8px;
        padding: 1rem 1.2rem;
        margin-bottom: 0.8rem;
    }
    
    .q-number {
        font-size: 0.7rem;
        color: #444444;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 0.4rem;
    }
    
    .q-text {
        font-size: 0.9rem;
        color: #e0e0e0;
        line-height: 1.5;
        margin-bottom: 0.6rem;
    }
    
    .q-option {
        font-size: 0.82rem;
        color: #888888;
        padding: 0.15rem 0;
    }
    
    .q-ref {
        font-size: 0.72rem;
        color: #444444;
        margin-top: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)


# ─── APP HEADER ──────────────────────────────────────────────────────────────

st.markdown('<div class="app-title">DocuMorph</div>', unsafe_allow_html=True)
st.markdown('<div class="app-subtitle">PDF to structured JSON — Gujarati MCQ extraction</div>', unsafe_allow_html=True)
st.markdown('<hr class="divider">', unsafe_allow_html=True)


# ─── FILE UPLOAD ─────────────────────────────────────────────────────────────

uploaded_file = st.file_uploader(
    "Upload PDF",
    type=["pdf"],
    help="Scanned Gujarati MCQ PDF"
)

# ─── SETTINGS ────────────────────────────────────────────────────────────────

with st.expander("Settings", expanded=False):
    col1, col2 = st.columns(2)
    
    with col1:
        engine = st.selectbox(
            "OCR Engine",
            ["groq", "tesseract"],
            index=0,
            help="Groq = high accuracy (needs API key). Tesseract = offline, lower accuracy."
        )
    
    with col2:
        page_range_input = st.text_input(
            "Page range (optional)",
            placeholder="e.g. 14-16",
            help="Leave empty to process all pages"
        )


# ─── PROCESS ─────────────────────────────────────────────────────────────────

if uploaded_file is not None:
    
    # Show file info
    file_size_mb = uploaded_file.size / (1024 * 1024)
    st.markdown(f"""
    <div class="status-box">
        <div class="status-label">File</div>
        <div class="status-value">{uploaded_file.name}</div>
        <div style="color: #444; font-size: 0.8rem; margin-top: 0.2rem;">{file_size_mb:.1f} MB</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Extract button
    if st.button("Extract Questions"):
        
        # Parse page range
        start_page = None
        end_page = None
        if page_range_input and page_range_input.strip():
            try:
                parts = page_range_input.strip().split('-')
                start_page = int(parts[0])
                end_page = int(parts[1]) if len(parts) > 1 else start_page
            except (ValueError, IndexError):
                st.error("Invalid page range. Use format like: 14-16")
                st.stop()
        
        # Save uploaded PDF to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.getvalue())
            tmp_path = tmp.name
        
        try:
            # Step 1: OCR
            status = st.empty()
            progress = st.progress(0)
            
            status.markdown(f"""
            <div class="status-box">
                <div class="status-label">Step 1 / 3</div>
                <div class="status-value">Extracting text from pages...</div>
            </div>
            """, unsafe_allow_html=True)
            progress.progress(10)
            
            if engine == "groq":
                pages_text = extract_text_groq(tmp_path, start_page, end_page)
            else:
                pages_text = extract_text_tesseract(tmp_path, start_page, end_page)
            
            if not pages_text:
                st.error("No text could be extracted from the PDF.")
                st.stop()
            
            progress.progress(50)
            
            # Step 2: Parse
            status.markdown(f"""
            <div class="status-box">
                <div class="status-label">Step 2 / 3</div>
                <div class="status-value">Parsing questions...</div>
            </div>
            """, unsafe_allow_html=True)
            
            questions = parse_questions(pages_text)
            progress.progress(80)
            
            if not questions:
                st.error("No questions found in the extracted text.")
                st.stop()
            
            # Step 3: Validate
            status.markdown(f"""
            <div class="status-box">
                <div class="status-label">Step 3 / 3</div>
                <div class="status-value">Validating output...</div>
            </div>
            """, unsafe_allow_html=True)
            
            # Build stats
            stats = {
                "total": len(questions),
                "with_4_options": sum(1 for q in questions if len(q.get("options", {})) == 4),
                "with_exam_ref": sum(1 for q in questions if "exam_reference" in q),
                "pages": len(set(q["page_number"] for q in questions)),
            }
            
            progress.progress(100)
            time.sleep(0.3)
            
            # Clear progress
            status.empty()
            progress.empty()
            
            # ─── RESULTS ────────────────────────────────────────────────
            
            st.markdown('<hr class="divider">', unsafe_allow_html=True)
            
            # Stats row
            c1, c2, c3, c4 = st.columns(4)
            
            with c1:
                st.markdown(f"""
                <div class="stat-card">
                    <div class="stat-number">{stats['total']}</div>
                    <div class="stat-label">Questions</div>
                </div>
                """, unsafe_allow_html=True)
            
            with c2:
                st.markdown(f"""
                <div class="stat-card">
                    <div class="stat-number">{stats['with_4_options']}</div>
                    <div class="stat-label">Complete</div>
                </div>
                """, unsafe_allow_html=True)
            
            with c3:
                st.markdown(f"""
                <div class="stat-card">
                    <div class="stat-number">{stats['with_exam_ref']}</div>
                    <div class="stat-label">With Ref</div>
                </div>
                """, unsafe_allow_html=True)
            
            with c4:
                st.markdown(f"""
                <div class="stat-card">
                    <div class="stat-number">{stats['pages']}</div>
                    <div class="stat-label">Pages</div>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("", unsafe_allow_html=True)
            
            # Build JSON output
            result_json = {
                "total_questions": len(questions),
                "questions": questions
            }
            json_str = json.dumps(result_json, ensure_ascii=False, indent=2)
            
            # Download button
            output_filename = f"{Path(uploaded_file.name).stem}_questions.json"
            
            st.download_button(
                label=f"Download JSON ({len(questions)} questions)",
                data=json_str.encode('utf-8'),
                file_name=output_filename,
                mime="application/json",
            )
            
            # Preview tabs
            tab1, tab2 = st.tabs(["Preview", "Raw JSON"])
            
            with tab1:
                # Show first 5 questions as cards
                show_count = min(5, len(questions))
                for q in questions[:show_count]:
                    options_html = ""
                    for key, val in q.get("options", {}).items():
                        options_html += f'<div class="q-option">({key}) {val}</div>'
                    
                    ref_html = ""
                    if "exam_reference" in q:
                        ref_html = f'<div class="q-ref">{q["exam_reference"]}</div>'
                    
                    st.markdown(f"""
                    <div class="q-card">
                        <div class="q-number">Q{q['question_number']} · Page {q['page_number']}</div>
                        <div class="q-text">{q['question_text'][:200]}{'...' if len(q['question_text']) > 200 else ''}</div>
                        {options_html}
                        {ref_html}
                    </div>
                    """, unsafe_allow_html=True)
                
                if len(questions) > show_count:
                    st.markdown(f'<div style="color: #444; font-size: 0.8rem; text-align: center; padding: 0.5rem;">+ {len(questions) - show_count} more questions in download</div>', unsafe_allow_html=True)
            
            with tab2:
                # Show raw JSON (truncated)
                preview_json = json_str[:3000]
                if len(json_str) > 3000:
                    preview_json += "\n\n... (truncated, download for full output)"
                st.markdown(f'<div class="json-preview">{preview_json}</div>', unsafe_allow_html=True)
        
        finally:
            # Clean up temp file
            try:
                os.unlink(tmp_path)
            except Exception:
                pass

else:
    # Empty state
    st.markdown("""
    <div style="text-align: center; padding: 3rem 1rem; color: #333333;">
        <div style="font-size: 2.5rem; margin-bottom: 0.5rem;">📄</div>
        <div style="font-size: 0.9rem;">Upload a scanned Gujarati PDF to begin</div>
    </div>
    """, unsafe_allow_html=True)
