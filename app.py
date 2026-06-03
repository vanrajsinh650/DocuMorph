import streamlit as st
import json
import tempfile
import sys
import traceback
from pathlib import Path


# Import core functions from main.py
from main import (
    extract_text_tesseract,
    enhance_pages_with_ai_robust,
    _get_page_count,
    parse_questions,
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
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─── CUSTOM CSS ──────────────────────────────────────────────────────────────

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    .stApp {
        background-color: #09090b;
        color: #a1a1aa;
    }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    .block-container {
        padding-top: 0rem !important;
        padding-bottom: 0rem !important;
        max-width: 100% !important;
        padding-left: 0 !important;
        padding-right: 0 !important;
    }

    /* Center and constrain all Streamlit blocks except the hero section */
    div[data-testid="element-container"]:not(:has(.hero)) {
        max-width: 900px !important;
        margin-left: auto !important;
        margin-right: auto !important;
        width: 100% !important;
    }

    /* Hide any default Streamlit or native hr elements that do not have our custom classes */
    hr:not(.section-divider):not(.divider) {
        display: none !important;
    }

    .stFileUploader, .stButton, .stDownloadButton, .stSelectbox, .stNumberInput, .stTextInput, .stExpander, .stTabs, div[data-testid="stHorizontalBlock"], div[data-testid="stExpander"] {
        max-width: 800px !important;
        margin-left: auto !important;
        margin-right: auto !important;
    }

    .status-box, .q-card, .resume-box, .divider, .stCodeBlock {
        max-width: 800px;
        margin-left: auto !important;
        margin-right: auto !important;
    }

    * {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    h1, h2, h3, h4, h5, h6 {
        color: #fafafa !important;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        font-weight: 600;
        letter-spacing: -0.025em;
    }

    p, span, label, .stMarkdown {
        color: #a1a1aa !important;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* ── HERO ── */
    .hero {
        position: relative;
        width: 100%;
        min-height: 70vh;
        background: linear-gradient(165deg, #0c1222 0%, #09090b 50%, #0b0d10 100%);
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        text-align: center;
        margin-bottom: 0;
        margin-top: -3rem;
        overflow: hidden;
    }
    .hero::before {
        content: '';
        position: absolute;
        top: -20%; left: 30%; width: 40%; height: 60%;
        background: radial-gradient(ellipse, rgba(56,189,248,0.08) 0%, transparent 70%);
        filter: blur(60px);
    }
    .hero::after {
        content: '';
        position: absolute;
        bottom: -10%; right: 20%; width: 30%; height: 40%;
        background: radial-gradient(ellipse, rgba(139,92,246,0.06) 0%, transparent 70%);
        filter: blur(50px);
    }
    .hero-content {
        position: relative; z-index: 1;
        max-width: 680px; padding: 2rem;
    }
    .hero-badge {
        display: inline-flex; align-items: center; gap: 0.375rem;
        background: rgba(56,189,248,0.1); border: 1px solid rgba(56,189,248,0.2);
        border-radius: 100px; padding: 0.25rem 0.75rem;
        font-size: 0.75rem; font-weight: 500; color: #38bdf8;
        margin-bottom: 1.5rem;
    }
    .hero-badge-dot {
        width: 6px; height: 6px; border-radius: 50%;
        background: #38bdf8;
        animation: pulse-dot 2s ease-in-out infinite;
    }
    @keyframes pulse-dot {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.4; }
    }
    .hero-title {
        font-size: 3.5rem; font-weight: 800; color: #ffffff;
        letter-spacing: -0.04em; margin-bottom: 1rem; line-height: 1.1;
    }
    .hero-subtitle {
        font-size: 1.0625rem; color: #a1a1aa;
        line-height: 1.7; max-width: 560px; margin: 0 auto;
    }
    .hero-subtitle strong { color: #e4e4e7; font-weight: 600; }

    /* ── SECTION WRAPPERS ── */
    .landing-section {
        max-width: 900px;
        margin: 0 auto !important;
        padding: 3rem 2rem;
    }
    .section-label {
        font-size: 0.6875rem; font-weight: 600; text-transform: uppercase;
        letter-spacing: 0.1em; color: #38bdf8; margin-bottom: 0.5rem;
    }
    .section-title {
        font-size: 1.75rem; font-weight: 700; color: #fafafa;
        letter-spacing: -0.03em; margin-bottom: 0.5rem;
    }
    .section-desc {
        font-size: 0.9375rem; color: #71717a; line-height: 1.6;
        max-width: 600px; margin-bottom: 2rem;
    }

    /* ── HOW IT WORKS PIPELINE ── */
    .pipeline {
        display: flex; align-items: flex-start; gap: 0;
        position: relative; margin: 1rem 0;
    }
    .pipe-step {
        flex: 1; display: flex; flex-direction: column; align-items: center;
        text-align: center; position: relative; padding: 0 0.5rem;
    }
    .pipe-icon {
        width: 56px; height: 56px; border-radius: 12px;
        background: #18181b; border: 1px solid #27272a;
        display: flex; align-items: center; justify-content: center;
        margin-bottom: 0.75rem; position: relative; z-index: 2;
        transition: border-color 0.2s, background-color 0.2s;
    }
    .pipe-icon svg { color: #71717a; }
    .pipe-step:hover .pipe-icon {
        border-color: #38bdf8; background: rgba(56,189,248,0.05);
    }
    .pipe-step:hover .pipe-icon svg { color: #38bdf8; }
    .pipe-num {
        position: absolute; top: -6px; right: -6px;
        width: 20px; height: 20px; border-radius: 50%;
        background: #27272a; border: 2px solid #09090b;
        display: flex; align-items: center; justify-content: center;
        font-size: 0.625rem; font-weight: 700; color: #a1a1aa;
    }
    .pipe-title {
        font-size: 0.8125rem; font-weight: 600; color: #fafafa;
        margin-bottom: 0.25rem;
    }
    .pipe-desc {
        font-size: 0.75rem; color: #71717a; line-height: 1.4;
        max-width: 140px;
    }
    .pipe-arrow {
        display: flex; align-items: center; padding-top: 1rem;
        color: #3f3f46; flex-shrink: 0;
    }

    /* ── FEATURES GRID ── */
    .features-grid {
        display: grid; grid-template-columns: repeat(3, 1fr);
        gap: 1rem; margin: 1rem 0;
    }
    @media (max-width: 640px) {
        .features-grid { grid-template-columns: 1fr; }
        .pipeline { flex-direction: column; align-items: center; }
        .pipe-arrow { transform: rotate(90deg); padding: 0.5rem 0; }
    }
    .feature-card {
        background: #18181b; border: 1px solid #27272a;
        border-radius: 8px; padding: 1.25rem;
        transition: border-color 0.2s;
    }
    .feature-card:hover { border-color: #3f3f46; }
    .feature-icon {
        width: 36px; height: 36px; border-radius: 8px;
        background: rgba(56,189,248,0.08);
        display: flex; align-items: center; justify-content: center;
        margin-bottom: 0.75rem;
    }
    .feature-icon svg { color: #38bdf8; }
    .feature-title {
        font-size: 0.875rem; font-weight: 600; color: #fafafa;
        margin-bottom: 0.25rem;
    }
    .feature-desc {
        font-size: 0.8125rem; color: #71717a; line-height: 1.5;
    }

    /* ── OUTPUT PREVIEW ── */
    .output-preview {
        background: #0f0f12; border: 1px solid #27272a;
        border-radius: 8px; overflow: hidden; margin: 1rem 0;
    }
    .output-tab-bar {
        display: flex; align-items: center; gap: 0;
        border-bottom: 1px solid #27272a;
        padding: 0 1rem; background: #18181b;
    }
    .output-tab {
        padding: 0.625rem 0.75rem; font-size: 0.75rem;
        color: #71717a; font-weight: 500;
        border-bottom: 2px solid transparent;
    }
    .output-tab.active {
        color: #fafafa; border-bottom-color: #38bdf8;
    }
    .output-body {
        padding: 1rem 1.25rem;
    }
    .output-body pre {
        margin: 0; font-size: 0.8125rem; color: #a1a1aa;
        line-height: 1.6; overflow-x: auto;
        font-family: 'JetBrains Mono', 'Fira Code', 'SF Mono', monospace;
    }
    .output-body .json-key { color: #38bdf8; }
    .output-body .json-str { color: #a78bfa; }
    .output-body .json-num { color: #fb923c; }

    /* ── SECTION DIVIDER ── */
    .section-divider {
        border: none; border-top: 1px solid #1a1a1e;
        margin: 0 auto !important;
        max-width: 900px !important;
        display: block !important;
    }

    /* ── FOOTER ── */
    .site-footer {
        max-width: 900px; margin: 0 auto !important;
        padding: 2rem 2rem; text-align: center;
        border-top: 1px solid #1a1a1e;
    }
    .footer-text {
        font-size: 0.75rem; color: #3f3f46;
    }
    .footer-text a { color: #71717a; text-decoration: none; }
    .footer-text a:hover { color: #a1a1aa; }

    /* ── TOOL WORKSPACE ELEMENTS (below hero when uploading) ── */
    .divider {
        border: none;
        border-top: 1px solid #27272a;
        margin: 1.5rem 0;
    }

    .stFileUploader > div {
        background-color: #09090b !important;
        border: 1px dashed #27272a !important;
        border-radius: 6px !important;
        transition: border-color 0.15s ease;
    }
    .stFileUploader > div:hover {
        border-color: #3f3f46 !important;
    }

    /* Remove Streamlit's bottom border/line under the file uploader */
    .stFileUploader {
        padding-bottom: 0 !important;
    }
    .stFileUploader > div > div {
        border-bottom: none !important;
    }
    div[data-testid="stFileUploader"] > section > div {
        border-bottom: none !important;
    }
    div[data-testid="stFileUploader"] + div > hr,
    div[data-testid="stFileUploader"] ~ hr {
        display: none !important;
    }
    /* Hide any small separator/line elements Streamlit injects */
    .stFileUploader small,
    .stFileUploader > div > small {
        display: none !important;
    }

    .stButton > button {
        background-color: #fafafa !important;
        color: #09090b !important;
        border: none !important;
        border-radius: 4px !important;
        font-weight: 500 !important;
        font-size: 0.875rem !important;
        padding: 0.5rem 1rem !important;
        width: 100%;
        transition: opacity 0.15s ease;
    }
    .stButton > button:hover { opacity: 0.9; }

    .stDownloadButton > button {
        background-color: #18181b !important;
        color: #fafafa !important;
        border: 1px solid #27272a !important;
        border-radius: 4px !important;
        font-weight: 500 !important;
        font-size: 0.875rem !important;
        width: 100%;
        transition: background-color 0.15s ease;
    }
    .stDownloadButton > button:hover {
        background-color: #27272a !important;
    }

    .stSelectbox > div > div,
    .stNumberInput > div > div > input,
    .stTextInput > div > div > input {
        background-color: #09090b !important;
        border: 1px solid #27272a !important;
        border-radius: 4px !important;
        color: #fafafa !important;
        font-size: 0.875rem !important;
    }

    .status-box {
        background-color: #18181b;
        border: 1px solid #27272a;
        border-radius: 6px;
        padding: 0.875rem 1rem;
        margin: 0.5rem 0;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .status-value {
        font-size: 0.875rem; color: #fafafa; font-weight: 500;
        display: flex; align-items: center; gap: 0.5rem;
    }
    .status-meta { font-size: 0.75rem; color: #71717a; }

    .stat-card {
        background-color: #09090b; border: 1px solid #27272a;
        border-radius: 6px; padding: 0.75rem 1rem;
        display: flex; flex-direction: column;
    }
    .stat-number {
        font-size: 1.25rem; font-weight: 600; color: #fafafa;
        margin-bottom: 0.125rem; line-height: 1;
    }
    .stat-label { font-size: 0.75rem; color: #71717a; font-weight: 500; }

    .stProgress > div > div { background-color: #27272a !important; }
    .stProgress > div > div > div { background-color: #fafafa !important; }

    .stTabs [data-baseweb="tab-list"] { background-color: transparent; gap: 1rem; }
    .stTabs [data-baseweb="tab"] {
        background-color: transparent !important;
        color: #71717a !important;
        border-bottom: 2px solid transparent;
        padding-bottom: 0.5rem; font-weight: 500;
        font-size: 0.875rem; margin-right: 1rem;
    }
    .stTabs [aria-selected="true"] {
        color: #fafafa !important;
        border-bottom-color: #fafafa !important;
    }

    .q-card {
        background-color: #18181b; border: 1px solid #27272a;
        border-radius: 6px; padding: 1rem; margin-bottom: 0.5rem;
    }
    .q-number { font-size: 0.75rem; color: #71717a; font-weight: 500; margin-bottom: 0.5rem; }
    .q-text { font-size: 0.875rem; color: #fafafa; line-height: 1.5; margin-bottom: 0.75rem; font-weight: 500; }
    .q-option { font-size: 0.875rem; color: #a1a1aa; padding: 0.125rem 0; display: flex; gap: 0.5rem; }
    .q-option-key { color: #71717a; font-weight: 500; }
    .q-ref { font-size: 0.75rem; color: #71717a; margin-top: 0.75rem; display: inline-block; background-color: #27272a; padding: 0.125rem 0.375rem; border-radius: 4px; }

    .stCodeBlock {
        background-color: #09090b !important;
        border: 1px solid #27272a !important;
        border-radius: 6px !important;
    }
    .stCodeBlock code {
        color: #a1a1aa !important;
        font-size: 0.75rem !important;
    }

    .resume-box {
        background-color: #18181b; border: 1px solid #b45309;
        border-radius: 6px; padding: 1rem; margin: 0.8rem 0;
    }
    .resume-title {
        font-size: 0.875rem; color: #fb923c; font-weight: 600;
        margin-bottom: 0.25rem;
        display: flex; align-items: center; gap: 0.5rem;
    }
    .resume-info { font-size: 0.8125rem; color: #a1a1aa; }
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


def _run_extraction(tmp_path: str, start_page: int, end_page: int,
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
        # Using standard Tesseract extraction
        engine = "tesseract"

        print(f"[1/3] Starting Base OCR — Tesseract")
        print(f"  Pages: {start_page} to {end_page}")

        progress_bar.progress(5)

        new_pages = extract_text_tesseract(tmp_path, start_page, end_page)

        if new_pages:
            pages_text.extend(new_pages)
            last_processed_page = end_page

        progress_bar.progress(50)
        print(f"[1/3] Base OCR complete — {len(new_pages)} pages extracted.")

        # Save raw pages_text in state so it can be enhanced later if the user chooses
        st.session_state["raw_pages_text"] = pages_text

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


def _compute_stats(questions: list[dict]) -> dict:
    return {
        "total": len(questions),
        "with_4_options": sum(1 for q in questions if len(q.get("options", {})) == 4),
        "with_exam_ref": sum(1 for q in questions if "exam_reference" in q),
        "pages": len(set(q["page_number"] for q in questions)) if questions else 0,
    }


def _run_extraction_robust(tmp_path: str, start_page: int, end_page: int,
                           log_container, progress_bar, existing_pages: list = None):
    """
    Robust extraction flow:
    1) Tesseract OCR
    2) Groq auto-correction
    3) Parse corrected output (fallback to raw parse)
    """
    logger = StreamlitLogCapture(log_container)
    old_stdout = sys.stdout
    sys.stdout = logger

    raw_pages_text = list(existing_pages) if existing_pages else []
    fixed_pages_text = []
    raw_questions = []
    questions = []
    stats = {}
    error_occurred = False
    last_processed_page = start_page - 1

    try:
        print(f"[1/4] Starting Base OCR - Tesseract")
        print(f"  Pages: {start_page} to {end_page}")
        progress_bar.progress(5)

        new_pages = extract_text_tesseract(tmp_path, start_page, end_page)
        if new_pages:
            raw_pages_text.extend(new_pages)
            last_processed_page = end_page

        progress_bar.progress(40)
        print(f"[1/4] Base OCR complete - {len(new_pages)} pages extracted.")
        st.session_state["raw_pages_text"] = raw_pages_text

        print(f"[2/4] Parsing raw Tesseract output...")
        raw_questions = parse_questions(raw_pages_text)
        print(f"[2/4] Parsed {len(raw_questions)} raw questions.")
        progress_bar.progress(55)

        print(f"[3/4] Auto-correcting Gujarati text with Groq...")
        checkpoint_path = f"{Path(tmp_path).stem}_streamlit_ai_checkpoint.json"
        fixed_pages_text = enhance_pages_with_ai_robust(
            raw_pages_text,
            ai_provider="groq",
            checkpoint_path=checkpoint_path,
            checkpoint_every=10,
        )
        st.session_state["fixed_pages_text"] = fixed_pages_text
        progress_bar.progress(80)

        print(f"[4/4] Parsing corrected output...")
        fixed_questions = parse_questions(fixed_pages_text)
        if fixed_questions:
            questions = fixed_questions
            print(f"[4/4] Parsed {len(fixed_questions)} corrected questions.")
        else:
            questions = raw_questions
            print("[4/4] Corrected parse empty, falling back to raw parsed questions.")

        stats = _compute_stats(questions)
        print(f"  Total: {stats['total']} questions")
        print(f"  Complete (4 options): {stats['with_4_options']}")
        print(f"  With exam ref: {stats['with_exam_ref']}")
        print(f"[4/4] Done.")
        progress_bar.progress(100)

    except Exception as e:
        error_occurred = True
        print(f"\n⚠ ERROR: {str(e)}")
        print(traceback.format_exc())

        if raw_pages_text:
            for p in reversed(raw_pages_text):
                if p.get("text"):
                    last_processed_page = p["page_number"]
                    break

            print(f"\n--- Partial results available ---")
            print(f"Pages extracted so far: {len(raw_pages_text)}")
            print(f"Last successful page: {last_processed_page}")

            try:
                raw_questions = parse_questions(raw_pages_text)
                questions = raw_questions
                stats = _compute_stats(questions)
                fixed_pages_text = raw_pages_text
                print(f"Parsed {len(questions)} questions from partial data.")
            except Exception:
                questions = []
                stats = {"total": 0, "with_4_options": 0, "with_exam_ref": 0, "pages": 0}

    finally:
        st.session_state["raw_pages_text"] = raw_pages_text
        st.session_state["fixed_pages_text"] = fixed_pages_text if fixed_pages_text else raw_pages_text
        st.session_state["raw_questions"] = raw_questions
        log_text = logger.get_logs()
        sys.stdout = old_stdout

    return raw_pages_text, questions, stats, log_text, last_processed_page, error_occurred


# ─── APP ─────────────────────────────────────────────────────────────────────

st.markdown("""
<div class="hero">
    <div class="hero-content">
        <div class="hero-badge"><span class="hero-badge-dot"></span> Open-Source OCR Pipeline</div>
        <div class="hero-title">DocuMorph</div>
        <div class="hero-subtitle">
            Extract <strong>Gujarati multiple-choice questions</strong> from scanned PDF exam papers
            and convert them into clean, <strong>structured JSON</strong> — automatically.
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader("Upload Document", type=["pdf"], help="Select a scanned Gujarati MCQ PDF to process", label_visibility="collapsed")

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
        for key in [
            "results",
            "partial_state",
            "raw_pages_text",
            "fixed_pages_text",
            "raw_questions",
            "ai_enhanced",
            "ai_log_msg",
        ]:
            if key in st.session_state:
                del st.session_state[key]

    total_pages = st.session_state.get("total_pages", 0)

    st.markdown(f"""
    <div class="status-box">
        <div class="status-value">
            <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="color: #a1a1aa"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path><polyline points="14 2 14 8 20 8"></polyline><line x1="16" y1="13" x2="8" y2="13"></line><line x1="16" y1="17" x2="8" y2="17"></line><polyline points="10 9 9 9 8 9"></polyline></svg>
            {uploaded_file.name}
        </div>
        <div class="status-meta">{file_size_mb:.1f} MB &bull; {total_pages} pages</div>
    </div>
    """, unsafe_allow_html=True)

    # ── Settings expander ──
    with st.expander("Settings", expanded=True):
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
            <div class="resume-title">
                <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"></path><line x1="12" y1="9" x2="12" y2="13"></line><line x1="12" y1="17" x2="12.01" y2="17"></line></svg>
                Previous run stopped
            </div>
            <div class="resume-info">
                Extracted {partial['pages_done']} pages &bull; {partial['questions_found']} questions found<br>
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
            pages_text, questions, stats, log_text, last_page, had_error = _run_extraction_robust(
                tmp_path, resume_start, resume_end,
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
                    raw_questions = st.session_state.get("raw_questions", [])
                    raw_json = {"total_questions": len(raw_questions), "questions": raw_questions}
                    raw_json_str = json.dumps(raw_json, ensure_ascii=False, indent=2)
                    st.session_state["results"] = {
                        "questions": questions,
                        "stats": stats,
                        "json_str": json_str,
                        "raw_json_str": raw_json_str,
                        "output_filename": f"{Path(uploaded_file.name).stem}_questions_fixed.json",
                        "raw_output_filename": f"{Path(uploaded_file.name).stem}_questions_raw.json",
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

            pages_text, questions, stats, log_text, last_page, had_error = _run_extraction_robust(
                tmp_path, start_page, end_page,
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
                    raw_questions = st.session_state.get("raw_questions", [])
                    raw_json = {"total_questions": len(raw_questions), "questions": raw_questions}
                    raw_json_str = json.dumps(raw_json, ensure_ascii=False, indent=2)
                    st.session_state["results"] = {
                        "questions": questions,
                        "stats": stats,
                        "json_str": json_str,
                        "raw_json_str": raw_json_str,
                        "output_filename": f"{Path(uploaded_file.name).stem}_questions_fixed.json",
                        "raw_output_filename": f"{Path(uploaded_file.name).stem}_questions_raw.json",
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
    raw_json_str = r.get("raw_json_str", json.dumps({"total_questions": 0, "questions": []}, ensure_ascii=False, indent=2))
    output_filename = r["output_filename"]
    raw_output_filename = r.get("raw_output_filename", f"{Path(uploaded_file.name).stem}_questions_raw.json")
    full_log = r["full_log"]

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    # --- AI Enhancement Box ---
    if False:  # Legacy manual enhancement flow (disabled)
        st.markdown("""
        <div style="background-color:#1e3a8a; padding: 1.5rem; border-radius: 8px; margin-bottom: 2rem; border-left: 4px solid #3b82f6;">
            <h3 style="margin-top: 0; color: #ffffff;">✨ Text Quality Looks Garbled?</h3>
            <p style="color: #bfdbfe; font-size: 0.95rem; margin-bottom: 1rem;">
                The initial extraction was generated using standard Tesseract OCR. If you notice strange letters or missing characters in the questions below, you can optionally run our AI Text Corrector to completely fix the text automatically.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("✨ Check & Fix Words with AI", type="primary", use_container_width=True):
            print("starting ai enhancement")
            # 1. Take raw text
            raw_pages_text = st.session_state["raw_pages_text"]

            # --- Live Log Setup ---
            st.markdown("**AI Enhancement Log**")
            ai_log_container = st.empty()

            # 2. Process with AI
            with st.spinner("AI is fixing Gujarati text... please wait..."):
                logger = StreamlitLogCapture(ai_log_container)
                old_stdout = sys.stdout
                sys.stdout = logger
                try:
                    enhanced_pages = enhance_pages_with_ai_robust(raw_pages_text, ai_provider="groq")
                finally:
                    sys.stdout = old_stdout
                    
                # 3. Save enhanced raw pages
                st.session_state["raw_pages_text"] = enhanced_pages
                # 4. Re-parse
                questions = parse_questions(enhanced_pages)
                
                stats = {
                    "total": len(questions),
                    "with_4_options": sum(1 for q in questions if len(q.get("options", {})) == 4),
                    "with_exam_ref": sum(1 for q in questions if "exam_reference" in q),
                    "pages": len(set(q["page_number"] for q in questions)),
                }
                
                result_json = {"total_questions": len(questions), "questions": questions}
                json_str = json.dumps(result_json, ensure_ascii=False, indent=2)
                
                st.session_state["results"] = {
                    "questions": questions,
                    "stats": stats,
                    "json_str": json_str,
                    "output_filename": output_filename,
                    "full_log": full_log,
                }
                st.session_state["ai_enhanced"] = True
                st.session_state["ai_log_msg"] = f"✅ AI successfully analyzed and corrected {len(enhanced_pages)} pages of text!"
                st.rerun()
                
    st.info("Auto-correction with Groq is already applied. Download both fixed and raw JSON below.")
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

    # Download buttons
    dc1, dc2 = st.columns(2)
    with dc1:
        st.download_button(
            label=f"Download Fixed JSON ({len(questions)} questions)",
            data=json_str.encode('utf-8'),
            file_name=output_filename,
            mime="application/json",
            use_container_width=True,
        )
    with dc2:
        raw_total = json.loads(raw_json_str).get("total_questions", 0)
        st.download_button(
            label=f"Download Raw JSON ({raw_total} questions)",
            data=raw_json_str.encode('utf-8'),
            file_name=raw_output_filename,
            mime="application/json",
            use_container_width=True,
        )

    # Preview tabs
    tab1, tab2, tab3, tab4 = st.tabs(["Preview", "Fixed JSON", "Raw JSON", "Full Log"])

    with tab1:
        show_count = min(5, len(questions))
        for q in questions[:show_count]:
            options_html = ""
            for key, val in q.get("options", {}).items():
                options_html += f'<div class="q-option"><span class="q-option-key">{key}</span> <span>{val}</span></div>'
            ref_html = ""
            if "exam_reference" in q:
                ref_html = f'<div class="q-ref">{q["exam_reference"]}</div>'

            st.markdown(f"""
            <div class="q-card">
                <div class="q-number">Q{q['question_number']} &bull; Page {q['page_number']}</div>
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
        preview_raw_json = raw_json_str[:5000]
        if len(raw_json_str) > 5000:
            preview_raw_json += "\n\n... (truncated, download for full output)"
        st.code(preview_raw_json, language="json")

    with tab4:
        st.code(full_log, language="text")

    # Clear results button
    if st.button("Clear Results", key="btn_clear"):
        for key in [
            "results",
            "partial_state",
            "raw_pages_text",
            "fixed_pages_text",
            "raw_questions",
            "ai_enhanced",
            "ai_log_msg",
        ]:
            if key in st.session_state:
                del st.session_state[key]
        st.rerun()

elif uploaded_file is None:
    # ── HOW IT WORKS ──
    st.markdown("""
    <hr class="section-divider">
    <div class="landing-section">
        <div class="section-label">How It Works</div>
        <div class="section-title">From scanned paper to structured data in 4 steps</div>
        <div class="section-desc">DocuMorph automates the entire extraction pipeline so you don't have to manually type out hundreds of questions from exam booklets.</div>

        <div class="pipeline">
            <div class="pipe-step">
                <div class="pipe-icon">
                    <span class="pipe-num">1</span>
                    <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="17 8 12 3 7 8"/><line x1="12" y1="3" x2="12" y2="15"/></svg>
                </div>
                <div class="pipe-title">Upload PDF</div>
                <div class="pipe-desc">Drop your scanned Gujarati MCQ exam paper</div>
            </div>

            <div class="pipe-arrow">
                <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><polyline points="9 18 15 12 9 6"/></svg>
            </div>

            <div class="pipe-step">
                <div class="pipe-icon">
                    <span class="pipe-num">2</span>
                    <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="3" width="18" height="18" rx="2" ry="2"/><circle cx="8.5" cy="8.5" r="1.5"/><polyline points="21 15 16 10 5 21"/></svg>
                </div>
                <div class="pipe-title">OCR Scan</div>
                <div class="pipe-desc">Tesseract reads each page with column-aware segmentation</div>
            </div>

            <div class="pipe-arrow">
                <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><polyline points="9 18 15 12 9 6"/></svg>
            </div>

            <div class="pipe-step">
                <div class="pipe-icon">
                    <span class="pipe-num">3</span>
                    <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><path d="M12 2L2 7l10 5 10-5-10-5z"/><path d="M2 17l10 5 10-5"/><path d="M2 12l10 5 10-5"/></svg>
                </div>
                <div class="pipe-title">AI Correction</div>
                <div class="pipe-desc">Groq LLM fixes garbled Gujarati characters automatically</div>
            </div>

            <div class="pipe-arrow">
                <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><polyline points="9 18 15 12 9 6"/></svg>
            </div>

            <div class="pipe-step">
                <div class="pipe-icon">
                    <span class="pipe-num">4</span>
                    <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><polyline points="16 18 22 12 16 6"/><polyline points="8 6 2 12 8 18"/></svg>
                </div>
                <div class="pipe-title">JSON Output</div>
                <div class="pipe-desc">Download structured question data ready for any app</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── FEATURES ──
    st.markdown("""
    <hr class="section-divider">
    <div class="landing-section">
        <div class="section-label">Features</div>
        <div class="section-title">Built for real-world exam papers</div>
        <div class="section-desc">Handles the messy reality of scanned documents — blurry text, dual columns, Indic script ligatures, and API rate limits.</div>

        <div class="features-grid">
            <div class="feature-card">
                <div class="feature-icon">
                    <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/></svg>
                </div>
                <div class="feature-title">Gujarati OCR</div>
                <div class="feature-desc">Tesseract + custom column segmentation tuned for Indic scripts and dual-column exam layouts.</div>
            </div>
            <div class="feature-card">
                <div class="feature-icon">
                    <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><path d="M12 2L2 7l10 5 10-5-10-5z"/><path d="M2 17l10 5 10-5"/><path d="M2 12l10 5 10-5"/></svg>
                </div>
                <div class="feature-title">AI Text Repair</div>
                <div class="feature-desc">Groq LLM auto-corrects garbled characters, broken ligatures, and OCR artefacts in Gujarati text.</div>
            </div>
            <div class="feature-card">
                <div class="feature-icon">
                    <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><rect x="2" y="3" width="20" height="14" rx="2" ry="2"/><line x1="8" y1="21" x2="16" y2="21"/><line x1="12" y1="17" x2="12" y2="21"/></svg>
                </div>
                <div class="feature-title">Batch Processing</div>
                <div class="feature-desc">Process hundreds of pages in configurable 100-page batches with automatic checkpointing.</div>
            </div>
            <div class="feature-card">
                <div class="feature-icon">
                    <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><polyline points="1 4 1 10 7 10"/><path d="M3.51 15a9 9 0 1 0 2.13-9.36L1 10"/></svg>
                </div>
                <div class="feature-title">Resume on Failure</div>
                <div class="feature-desc">If processing stops mid-way, pick up exactly where you left off without re-doing finished pages.</div>
            </div>
            <div class="feature-card">
                <div class="feature-icon">
                    <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/></svg>
                </div>
                <div class="feature-title">Rate-Limit Safe</div>
                <div class="feature-desc">Multi-key API pool with round-robin rotation, adaptive backoff, and jitter to avoid 429 errors.</div>
            </div>
            <div class="feature-card">
                <div class="feature-icon">
                    <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/><line x1="16" y1="13" x2="8" y2="13"/><line x1="16" y1="17" x2="8" y2="17"/></svg>
                </div>
                <div class="feature-title">Dual JSON Output</div>
                <div class="feature-desc">Download both raw and AI-corrected JSON so you can compare or use whichever suits your needs.</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── OUTPUT SCHEMA PREVIEW ──
    st.markdown("""
    <hr class="section-divider">
    <div class="landing-section">
        <div class="section-label">Output Format</div>
        <div class="section-title">Clean, structured JSON — ready for any application</div>
        <div class="section-desc">Each question is extracted with its number, full text, all options, exam reference, and source page number.</div>

        <div class="output-preview">
            <div class="output-tab-bar">
                <div class="output-tab active">output.json</div>
            </div>
            <div class="output-body">
<pre>{
  <span class="json-key">"total_questions"</span>: <span class="json-num">156</span>,
  <span class="json-key">"questions"</span>: [
    {
      <span class="json-key">"id"</span>: <span class="json-num">1</span>,
      <span class="json-key">"question_number"</span>: <span class="json-str">"001"</span>,
      <span class="json-key">"question_text"</span>: <span class="json-str">"ગુજરાતી ભાષાનો પ્રશ્ન..."</span>,
      <span class="json-key">"options"</span>: {
        <span class="json-key">"A"</span>: <span class="json-str">"વિકલ્પ એક"</span>,
        <span class="json-key">"B"</span>: <span class="json-str">"વિકલ્પ બે"</span>,
        <span class="json-key">"C"</span>: <span class="json-str">"વિકલ્પ ત્રણ"</span>,
        <span class="json-key">"D"</span>: <span class="json-str">"વિકલ્પ ચાર"</span>
      },
      <span class="json-key">"exam_reference"</span>: <span class="json-str">"PI 38/2017-18"</span>,
      <span class="json-key">"page_number"</span>: <span class="json-num">1</span>
    }
  ]
}</pre>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── FOOTER ──
    st.markdown("""
    <div class="site-footer">
        <div class="footer-text">
            Built by <a href="https://github.com/vanrajsinh650" target="_blank">vanrajsinh650</a> &bull;
            <a href="https://github.com/vanrajsinh650/DocuMorph" target="_blank">View on GitHub</a>
        </div>
    </div>
    """, unsafe_allow_html=True)
