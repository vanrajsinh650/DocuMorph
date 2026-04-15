import streamlit as st
import json
import tempfile
import os
import sys
import time
import math
import traceback
from pathlib import Path

# Import core functions from main.py
from main import (
    GroqKeyPool,
    OpenRouterKeyPool,
    extract_text_groq,
    extract_text_tesseract,
    extract_text_openrouter,
    extract_text_tesseract_ai,
    _get_page_count,
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
            try:
                self.log_container.code(log_text, language="text")
            except Exception:
                pass
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

    .stTabs [data-baseweb="tab-list"] { background-color: transparent; gap: 2rem; }
    .stTabs [data-baseweb="tab"] {
        background-color: transparent !important;
        color: #666666 !important;
        border-bottom: 2px solid transparent;
        padding-bottom: 0.5rem;
        font-weight: 500;
        font-size: 1rem;
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

    .resume-box {
        background-color: #111111;
        border: 1px solid #2a2a1a;
        border-radius: 8px;
        padding: 1rem 1.2rem;
        margin: 0.8rem 0;
    }
    .resume-title {
        font-size: 0.85rem;
        color: #ccaa44;
        font-weight: 600;
        margin-bottom: 0.4rem;
    }
    .resume-info {
        font-size: 0.8rem;
        color: #888888;
    }
</style>
""", unsafe_allow_html=True)


# ─── HELPERS ─────────────────────────────────────────────────────────────────

def _build_batch_options(total_pages: int, batch_size: int = 100) -> list[str]:
    """Build batch page range options like '1–100', '101–200', etc."""
    options = []
    for start in range(1, total_pages + 1, batch_size):
        end = min(start + batch_size - 1, total_pages)
        options.append(f"{start}–{end}")
    return options


def _parse_range_str(range_str: str) -> tuple[int, int]:
    """Parse a range string like '101–200' or '14-16' into (start, end)."""
    # Handle both en-dash and hyphen
    for sep in ["–", "-"]:
        if sep in range_str:
            parts = range_str.split(sep)
            return int(parts[0].strip()), int(parts[1].strip())
    # Single number
    val = int(range_str.strip())
    return val, val


def _run_extraction(tmp_path: str, engine: str, start_page: int, end_page: int,
                    log_container, progress_bar, existing_pages: list = None):
    """
    Run OCR extraction for the given page range.
    Returns (pages_text, questions, stats, log_text, last_processed_page, error_occurred).
    If an error occurs mid-way, returns partial results.
    """
    logger = StreamlitLogCapture(log_container)
    old_stdout = sys.stdout
    sys.stdout = logger

    pages_text = list(existing_pages) if existing_pages else []
    questions = []
    stats = {}
    error_occurred = False
    last_processed_page = start_page - 1

    try:
        # Step 1: OCR
        engine_label = {
            "openrouter": "OpenRouter (Qwen-2.5-VL)",
            "groq": "Groq Vision (Llama 4)",
            "tesseract": "Tesseract OCR",
            "tesseract+ai": "Tesseract + AI Fix (Hybrid)",
        }.get(engine, engine)

        print(f"[1/3] Starting OCR — {engine_label}")
        print(f"  Pages: {start_page} to {end_page}")

        progress_bar.progress(5)

        if engine == "openrouter":
            new_pages = extract_text_openrouter(tmp_path, start_page, end_page)
        elif engine == "groq":
            new_pages = extract_text_groq(tmp_path, start_page, end_page)
        elif engine == "tesseract+ai":
            new_pages = extract_text_tesseract_ai(tmp_path, start_page, end_page, ai_provider="openrouter")
        else:
            new_pages = extract_text_tesseract(tmp_path, start_page, end_page)

        if new_pages:
            pages_text.extend(new_pages)
            last_processed_page = end_page

        progress_bar.progress(50)
        print(f"[1/3] OCR complete — {len(new_pages)} pages extracted.")

        # Step 2: Parse
        print(f"[2/3] Parsing questions...")
        questions = parse_questions(pages_text)
        progress_bar.progress(80)
        print(f"[2/3] Parsed {len(questions)} questions.")

        # Step 3: Stats
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
        print(f"[3/3] Done.")
        progress_bar.progress(100)

    except Exception as e:
        error_occurred = True
        print(f"\n⚠ ERROR: {str(e)}")
        print(traceback.format_exc())

        # Try to figure out what we got so far
        if pages_text:
            # Find last page with actual text
            for p in reversed(pages_text):
                if p.get("text"):
                    last_processed_page = p["page_number"]
                    break

            print(f"\n--- Partial results available ---")
            print(f"Pages extracted so far: {len(pages_text)}")
            print(f"Last successful page: {last_processed_page}")

            # Try parsing what we have
            try:
                questions = parse_questions(pages_text)
                stats = {
                    "total": len(questions),
                    "with_4_options": sum(1 for q in questions if len(q.get("options", {})) == 4),
                    "with_exam_ref": sum(1 for q in questions if "exam_reference" in q),
                    "pages": len(set(q["page_number"] for q in questions)),
                }
                print(f"Parsed {len(questions)} questions from partial data.")
            except Exception:
                questions = []
                stats = {"total": 0, "with_4_options": 0, "with_exam_ref": 0, "pages": 0}

    finally:
        log_text = logger.get_logs()
        sys.stdout = old_stdout

    return pages_text, questions, stats, log_text, last_processed_page, error_occurred


# ─── APP ─────────────────────────────────────────────────────────────────────

st.markdown('<div class="app-title">DocuMorph</div>', unsafe_allow_html=True)
st.markdown('<div class="app-subtitle">PDF to structured JSON — Gujarati MCQ extraction</div>', unsafe_allow_html=True)
st.markdown('<hr class="divider">', unsafe_allow_html=True)

uploaded_file = st.file_uploader("Upload PDF", type=["pdf"], help="Scanned Gujarati MCQ PDF")

# ─── SETTINGS ────────────────────────────────────────────────────────────────

if uploaded_file is not None:

    file_size_mb = uploaded_file.size / (1024 * 1024)

    # Save temp file to detect page count
    if "tmp_path" not in st.session_state or st.session_state.get("_uploaded_name") != uploaded_file.name:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.getvalue())
            st.session_state["tmp_path"] = tmp.name
            st.session_state["_uploaded_name"] = uploaded_file.name
        total_pages = _get_page_count(st.session_state["tmp_path"])
        st.session_state["total_pages"] = total_pages
        # Clear old results when a new file is uploaded
        for key in ["results", "partial_state"]:
            if key in st.session_state:
                del st.session_state[key]

    total_pages = st.session_state.get("total_pages", 0)

    st.markdown(f"""
    <div class="status-box">
        <div class="status-label">File</div>
        <div class="status-value">{uploaded_file.name}</div>
        <div style="color: #444; font-size: 0.8rem; margin-top: 0.2rem;">{file_size_mb:.1f} MB · {total_pages} pages</div>
    </div>
    """, unsafe_allow_html=True)

    # ── Settings expander ──
    with st.expander("Settings", expanded=True):
        # Using cheap text correction hybrid pipeline
        engine = "tesseract+ai"

        st.markdown("**Page Selection**")
        page_mode = st.radio(
            "How to process pages",
            ["All pages", "Batch (100-page chunks)", "Custom range"],
            index=1 if total_pages > 100 else 0,
            horizontal=True,
            label_visibility="collapsed",
        )

        start_page = 1
        end_page = total_pages

        if page_mode == "Batch (100-page chunks)":
            batches = _build_batch_options(total_pages, batch_size=100)
            selected_batch = st.selectbox("Select batch", batches, index=0)
            start_page, end_page = _parse_range_str(selected_batch)

        elif page_mode == "Custom range":
            col1, col2 = st.columns(2)
            with col1:
                start_page = st.number_input("From page", min_value=1, max_value=total_pages, value=1, step=1)
            with col2:
                end_page = st.number_input("To page", min_value=1, max_value=total_pages, value=min(total_pages, 100), step=1)

            if start_page > end_page:
                st.error("'From page' must be ≤ 'To page'")
                st.stop()

        pages_to_process = end_page - start_page + 1
        st.markdown(f'<div style="color: #666; font-size: 0.8rem;">Will process **{pages_to_process}** pages ({start_page} → {end_page})</div>', unsafe_allow_html=True)


    # ─── RESUME STATE ────────────────────────────────────────────────────────

    partial = st.session_state.get("partial_state")

    if partial and not st.session_state.get("results"):
        st.markdown(f"""
        <div class="resume-box">
            <div class="resume-title">⚠ Previous run stopped</div>
            <div class="resume-info">
                Extracted {partial['pages_done']} pages · {partial['questions_found']} questions found<br>
                Last page processed: {partial['last_page']}<br>
                Original range: {partial['original_start']}–{partial['original_end']}
            </div>
        </div>
        """, unsafe_allow_html=True)

        rc1, rc2, rc3 = st.columns(3)
        with rc1:
            resume_clicked = st.button("▶ Resume", key="btn_resume")
        with rc2:
            download_partial = st.button("💾 Download Partial", key="btn_partial_dl")
        with rc3:
            reset_clicked = st.button("🔄 Reset", key="btn_reset")

        if reset_clicked:
            del st.session_state["partial_state"]
            st.rerun()

        if download_partial:
            pq = partial.get("questions", [])
            pj = json.dumps({"total_questions": len(pq), "questions": pq}, ensure_ascii=False, indent=2)
            st.download_button(
                label=f"Download partial ({len(pq)} questions)",
                data=pj.encode("utf-8"),
                file_name=f"{Path(uploaded_file.name).stem}_partial.json",
                mime="application/json",
                key="dl_partial"
            )

        if resume_clicked:
            # Continue from where it stopped
            resume_start = partial["last_page"] + 1
            resume_end = partial["original_end"]

            if resume_start > resume_end:
                st.warning("All pages were already processed. Download the partial results.")
                st.stop()

            st.markdown('<hr class="divider">', unsafe_allow_html=True)
            st.markdown(f"**Resuming** from page {resume_start} to {resume_end}")
            log_container = st.empty()
            progress = st.progress(0)

            tmp_path = st.session_state["tmp_path"]
            pages_text, questions, stats, log_text, last_page, had_error = _run_extraction(
                tmp_path, engine, resume_start, resume_end,
                log_container, progress,
                existing_pages=partial.get("pages_text", [])
            )

            if had_error and pages_text:
                # Save updated partial state
                st.session_state["partial_state"] = {
                    "pages_text": pages_text,
                    "questions": questions,
                    "questions_found": len(questions),
                    "pages_done": len(pages_text),
                    "last_page": last_page,
                    "original_start": partial["original_start"],
                    "original_end": resume_end,
                    "log": log_text,
                }
                st.warning(f"Stopped again at page {last_page}. {len(questions)} questions extracted so far.")
                st.rerun()
            else:
                # Finished successfully
                if "partial_state" in st.session_state:
                    del st.session_state["partial_state"]

                if questions:
                    result_json = {"total_questions": len(questions), "questions": questions}
                    json_str = json.dumps(result_json, ensure_ascii=False, indent=2)
                    st.session_state["results"] = {
                        "questions": questions,
                        "stats": stats,
                        "json_str": json_str,
                        "output_filename": f"{Path(uploaded_file.name).stem}_questions.json",
                        "full_log": log_text,
                    }
                    st.rerun()
                else:
                    st.error("No questions found after resume.")


    # ─── EXTRACT BUTTON ──────────────────────────────────────────────────────

    if not partial or st.session_state.get("results"):
        if st.button("Extract Questions", key="btn_extract"):

            st.markdown('<hr class="divider">', unsafe_allow_html=True)
            st.markdown("**Log**")
            log_container = st.empty()
            progress = st.progress(0)

            tmp_path = st.session_state["tmp_path"]

            pages_text, questions, stats, log_text, last_page, had_error = _run_extraction(
                tmp_path, engine, start_page, end_page,
                log_container, progress
            )

            if had_error and pages_text:
                # Save partial state for resume
                st.session_state["partial_state"] = {
                    "pages_text": pages_text,
                    "questions": questions,
                    "questions_found": len(questions),
                    "pages_done": len(pages_text),
                    "last_page": last_page,
                    "original_start": start_page,
                    "original_end": end_page,
                    "log": log_text,
                }
                st.warning(f"Processing stopped at page {last_page}. {len(questions)} questions extracted so far.")
                st.rerun()

            elif had_error and not pages_text:
                st.error("Failed to extract any pages. Check the log above.")

            else:
                # Full success
                if "partial_state" in st.session_state:
                    del st.session_state["partial_state"]

                if questions:
                    result_json = {"total_questions": len(questions), "questions": questions}
                    json_str = json.dumps(result_json, ensure_ascii=False, indent=2)
                    st.session_state["results"] = {
                        "questions": questions,
                        "stats": stats,
                        "json_str": json_str,
                        "output_filename": f"{Path(uploaded_file.name).stem}_questions.json",
                        "full_log": log_text,
                    }
                    st.rerun()
                else:
                    st.error("No questions found in the extracted text.")


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
    if st.button("Clear Results", key="btn_clear"):
        for key in ["results", "partial_state"]:
            if key in st.session_state:
                del st.session_state[key]
        st.rerun()

elif uploaded_file is None:
    st.markdown("""
    <div style="text-align: center; padding: 3rem 1rem; color: #333333;">
        <div style="font-size: 2.5rem; margin-bottom: 0.5rem;">📄</div>
        <div style="font-size: 0.9rem;">Upload a scanned Gujarati PDF to begin</div>
    </div>
    """, unsafe_allow_html=True)
