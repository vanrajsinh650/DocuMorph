import streamlit as st
import json
import tempfile
import os
import sys
import time
import io
import traceback
from pathlib import Path
from contextlib import redirect_stdout, redirect_stderr

# Import core functions from main.py
from main import (
    GroqKeyPool,
    extract_text_groq,
    extract_text_tesseract,
    parse_questions,
    validate_questions,
    save_raw_text,
)


# ─── LOG CAPTURE ─────────────────────────────────────────────────────────────

class StreamlitLogCapture:
    """Captures print() output and displays it live in a Streamlit container."""
    
    def __init__(self, log_container):
        self.log_container = log_container
        self.logs = []
        self._original_stdout = sys.stdout
    
    def write(self, text):
        if text.strip():
            self.logs.append(text.strip())
            # Update the log display (show last 30 lines)
            visible = self.logs[-30:]
            log_text = "\n".join(visible)
            self.log_container.code(log_text, language="text")
        # Also write to original stdout for terminal
        if self._original_stdout:
            try:
                self._original_stdout.write(text)
            except Exception:
                pass
    
    def flush(self):
        if self._original_stdout:
            try:
                self._original_stdout.flush()
            except Exception:
                pass
    
    def get_logs(self):
        return "\n".join(self.logs)


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
    .stApp {
        background-color: #0a0a0a;
        color: #e0e0e0;
    }
    
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 720px;
    }
    
    h1, h2, h3, h4, h5, h6 {
        color: #ffffff !important;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        font-weight: 600;
    }
    
    p, span, label, .stMarkdown {
        color: #b0b0b0 !important;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
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
    
    .divider {
        border: none;
        border-top: 1px solid #1a1a1a;
        margin: 1.5rem 0;
    }
    
    .stFileUploader > div {
        background-color: #111111 !important;
        border: 1px solid #222222 !important;
        border-radius: 8px !important;
    }
    
    .stButton > button {
        background-color: #ffffff !important;
        color: #000000 !important;
        border: none !important;
        border-radius: 6px !important;
        font-weight: 600 !important;
        font-size: 0.85rem !important;
        padding: 0.5rem 1.5rem !important;
        width: 100%;
    }
    
    .stButton > button:hover { opacity: 0.85; }
    
    .stDownloadButton > button {
        background-color: #ffffff !important;
        color: #000000 !important;
        border: none !important;
        border-radius: 6px !important;
        font-weight: 600 !important;
        width: 100%;
    }
    
    .stSelectbox > div > div,
    .stNumberInput > div > div > input,
    .stTextInput > div > div > input {
        background-color: #111111 !important;
        border-color: #222222 !important;
        color: #e0e0e0 !important;
    }

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
    
    .stProgress > div > div { background-color: #222222 !important; }
    .stProgress > div > div > div { background-color: #ffffff !important; }
    
    .stTabs [data-baseweb="tab-list"] { background-color: transparent; gap: 0; }
    .stTabs [data-baseweb="tab"] {
        background-color: transparent !important;
        color: #666666 !important;
        border-bottom: 2px solid transparent;
    }
    .stTabs [aria-selected="true"] {
        color: #ffffff !important;
        border-bottom-color: #ffffff !important;
    }
    
    .q-card {
        background-color: #111111;
        border: 1px solid #1a1a1a;
        border-radius: 8px;
        padding: 1rem 1.2rem;
        margin-bottom: 0.8rem;
    }
    .q-number { font-size: 0.7rem; color: #444444; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 0.4rem; }
    .q-text { font-size: 0.9rem; color: #e0e0e0; line-height: 1.5; margin-bottom: 0.6rem; }
    .q-option { font-size: 0.82rem; color: #888888; padding: 0.15rem 0; }
    .q-ref { font-size: 0.72rem; color: #444444; margin-top: 0.5rem; }
    
    /* Log panel */
    .stCodeBlock {
        background-color: #0d0d0d !important;
        border: 1px solid #1a1a1a !important;
    }
    .stCodeBlock code {
        color: #888888 !important;
        font-size: 0.75rem !important;
    }
</style>
""", unsafe_allow_html=True)


# ─── APP ─────────────────────────────────────────────────────────────────────

st.markdown('<div class="app-title">DocuMorph</div>', unsafe_allow_html=True)
st.markdown('<div class="app-subtitle">PDF to structured JSON — Gujarati MCQ extraction</div>', unsafe_allow_html=True)
st.markdown('<hr class="divider">', unsafe_allow_html=True)

uploaded_file = st.file_uploader("Upload PDF", type=["pdf"], help="Scanned Gujarati MCQ PDF")

with st.expander("Settings", expanded=False):
    col1, col2 = st.columns(2)
    with col1:
        engine = st.selectbox("OCR Engine", ["groq", "tesseract"], index=0)
    with col2:
        page_range_input = st.text_input("Page range (optional)", placeholder="e.g. 14-16")


# ─── PROCESS ─────────────────────────────────────────────────────────────────

if uploaded_file is not None:
    
    file_size_mb = uploaded_file.size / (1024 * 1024)
    st.markdown(f"""
    <div class="status-box">
        <div class="status-label">File</div>
        <div class="status-value">{uploaded_file.name}</div>
        <div style="color: #444; font-size: 0.8rem; margin-top: 0.2rem;">{file_size_mb:.1f} MB</div>
    </div>
    """, unsafe_allow_html=True)
    
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
        
        # Save to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.getvalue())
            tmp_path = tmp.name
        
        try:
            # Create log panel
            st.markdown('<hr class="divider">', unsafe_allow_html=True)
            st.markdown("**Log**")
            log_container = st.empty()
            progress = st.progress(0)
            
            # Set up log capture
            logger = StreamlitLogCapture(log_container)
            old_stdout = sys.stdout
            sys.stdout = logger
            
            try:
                # Step 1: OCR
                print(f"[1/3] Starting OCR ({engine} engine)...")
                print(f"  PDF: {uploaded_file.name}")
                if start_page:
                    print(f"  Pages: {start_page} to {end_page}")
                else:
                    print(f"  Pages: ALL")
                
                progress.progress(5)
                
                if engine == "groq":
                    pages_text = extract_text_groq(tmp_path, start_page, end_page)
                else:
                    pages_text = extract_text_tesseract(tmp_path, start_page, end_page)
                
                if not pages_text:
                    print("ERROR: No text extracted from PDF.")
                    sys.stdout = old_stdout
                    st.error("No text could be extracted from the PDF.")
                    st.stop()
                
                progress.progress(50)
                print(f"[1/3] OCR complete. Extracted text from {len(pages_text)} pages.")
                
                # Step 2: Parse
                print(f"[2/3] Parsing questions...")
                questions = parse_questions(pages_text)
                progress.progress(80)
                
                if not questions:
                    print("ERROR: No questions found in the text.")
                    sys.stdout = old_stdout
                    st.error("No questions found in the extracted text.")
                    st.stop()
                
                print(f"[2/3] Parsed {len(questions)} questions.")
                
                # Step 3: Validate
                print(f"[3/3] Validating...")
                stats = {
                    "total": len(questions),
                    "with_4_options": sum(1 for q in questions if len(q.get("options", {})) == 4),
                    "with_exam_ref": sum(1 for q in questions if "exam_reference" in q),
                    "pages": len(set(q["page_number"] for q in questions)),
                }
                print(f"  Total: {stats['total']} questions")
                print(f"  Complete (4 options): {stats['with_4_options']}")
                print(f"  With exam ref: {stats['with_exam_ref']}")
                print(f"  Pages: {stats['pages']}")
                print(f"[3/3] Done.")
                
                progress.progress(100)
                
                # Build JSON
                result_json = {
                    "total_questions": len(questions),
                    "questions": questions
                }
                json_str = json.dumps(result_json, ensure_ascii=False, indent=2)
                output_filename = f"{Path(uploaded_file.name).stem}_questions.json"
                
                # Store results in session_state so they persist across reruns
                st.session_state["results"] = {
                    "questions": questions,
                    "stats": stats,
                    "json_str": json_str,
                    "output_filename": output_filename,
                    "full_log": logger.get_logs(),
                }
                
            except Exception as e:
                print(f"ERROR: {str(e)}")
                print(traceback.format_exc())
                sys.stdout = old_stdout
                st.error(f"Error: {str(e)}")
                st.stop()
            finally:
                full_log = logger.get_logs()
                sys.stdout = old_stdout
        
        finally:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass

# ─── SHOW RESULTS (persisted in session_state) ──────────────────────────────

if "results" in st.session_state:
    r = st.session_state["results"]
    questions = r["questions"]
    stats = r["stats"]
    json_str = r["json_str"]
    output_filename = r["output_filename"]
    full_log = r["full_log"]
    
    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    
    # Stats row
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f'<div class="stat-card"><div class="stat-number">{stats["total"]}</div><div class="stat-label">Questions</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="stat-card"><div class="stat-number">{stats["with_4_options"]}</div><div class="stat-label">Complete</div></div>', unsafe_allow_html=True)
    with c3:
        st.markdown(f'<div class="stat-card"><div class="stat-number">{stats["with_exam_ref"]}</div><div class="stat-label">With Ref</div></div>', unsafe_allow_html=True)
    with c4:
        st.markdown(f'<div class="stat-card"><div class="stat-number">{stats["pages"]}</div><div class="stat-label">Pages</div></div>', unsafe_allow_html=True)
    
    st.markdown("", unsafe_allow_html=True)
    
    # Download button
    st.download_button(
        label=f"Download JSON ({len(questions)} questions)",
        data=json_str.encode('utf-8'),
        file_name=output_filename,
        mime="application/json",
    )
    
    # Preview tabs
    tab1, tab2, tab3 = st.tabs(["Preview", "Raw JSON", "Full Log"])
    
    with tab1:
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
            st.markdown(f'<div style="color: #444; font-size: 0.8rem; text-align: center;">+ {len(questions) - show_count} more in download</div>', unsafe_allow_html=True)
    
    with tab2:
        preview_json = json_str[:5000]
        if len(json_str) > 5000:
            preview_json += "\n\n... (truncated, download for full output)"
        st.code(preview_json, language="json")
    
    with tab3:
        st.code(full_log, language="text")
    
    # Clear results button
    if st.button("Clear Results"):
        del st.session_state["results"]
        st.rerun()

elif uploaded_file is None:
    st.markdown("""
    <div style="text-align: center; padding: 3rem 1rem; color: #333333;">
        <div style="font-size: 2.5rem; margin-bottom: 0.5rem;">📄</div>
        <div style="font-size: 0.9rem;">Upload a scanned Gujarati PDF to begin</div>
    </div>
    """, unsafe_allow_html=True)

