import sys
import io

# Fix Windows console encoding for Gujarati/Unicode output (only when running directly)
if __name__ == "__main__":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace', line_buffering=True)

import base64
import re
import json
import os
import time
import random
import platform
import argparse
from pathlib import Path
from io import BytesIO

from PIL import Image, ImageFilter, ImageEnhance
import fitz  # PyMuPDF
import gc

# CONFIGURATION (auto-detect platform)
if platform.system() == "Windows":
    TESSERACT_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
else:
    # Linux (Streamlit Cloud) — tesseract is a system package
    TESSERACT_PATH = "tesseract"

# Groq API config
GROQ_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"

# OpenRouter API config — model cascade (tries in order, falls back on 402/failure)
OPENROUTER_MODELS = [
    "qwen/qwen3-vl-8b-instruct",           # 1st: cheapest paid ($0.08/M), fast
    "qwen/qwen2.5-vl-72b-instruct",         # 2nd: best quality ($0.80/M)
    "google/gemma-3-27b-it:free",           # 3rd: FREE fallback (rate-limited but $0 forever)
]
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# OpenRouter TEXT-ONLY models for AI text correction (no vision needed, much cheaper)
OPENROUTER_TEXT_MODELS = [
    "qwen/qwen-2.5-72b-instruct",           # 1st: extremely smart for formatting & Indic languages
    "google/gemma-3-27b-it:free",            # 2nd: FREE fallback
]

# Groq TEXT model for AI text correction
GROQ_TEXT_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"

# Groq reliability tuning (reliability-first defaults)
GROQ_MIN_REQUEST_INTERVAL_SECONDS = 0.35
GROQ_BASE_429_COOLDOWN_SECONDS = 8
GROQ_MAX_COOLDOWN_SECONDS = 300
GROQ_PAGE_MAX_WAIT_SECONDS = 240
GROQ_PAGE_MAX_ATTEMPTS = 24
GROQ_TEXT_MAX_WAIT_SECONDS = 180
GROQ_TEXT_MAX_ATTEMPTS = 20
AI_CORRECTION_MIN_NONSPACE_CHARS = 20

# The prompt that instructs the vision model to extract text accurately
GROQ_OCR_PROMPT = """You are an expert OCR system for Gujarati language documents. Extract ALL text from this image EXACTLY as it appears in the document.

CRITICAL RULES:
1. This page has a TWO-COLUMN layout. Read the LEFT column first (top to bottom), then the RIGHT column (top to bottom).
2. Extract EVERY Gujarati word EXACTLY as written - do NOT transliterate, translate, or modify any word.
3. Preserve the exact question number format (e.g., 001), 002), etc.)
4. Preserve exam references in parentheses like (PI 38/2017-18), (STI 139/2020-21), etc.
5. Preserve option labels: (A), (B), (C), (D) exactly as they appear.
6. Keep all Gujarati characters, matras, and conjuncts EXACTLY as they appear in the original.
7. Do NOT add any commentary, notes, or explanations. Output ONLY the extracted text.
8. Ignore any answer keys at the bottom of columns (like "1-C/2-D/3-A..." patterns).
9. Ignore any watermarks or page headers/footers.

Output the raw text only, maintaining the original line structure as much as possible."""

# Prompt for AI text correction of Tesseract OCR output
AI_TEXT_FIX_PROMPT = """You are a Gujarati text correction expert. Below is OCR output from a scanned Gujarati MCQ exam book, extracted by Tesseract OCR. The text has errors — some Gujarati words are garbled, partially replaced with random English characters, or incorrectly recognized.

Your job is to FIX the broken/garbled words while preserving everything that is already correct.
The text is from a GPSC History/Culture Exam. You will encounter names like અશોક, સ્કંદગુપ્ત, ખારવેલ, રુદ્રદમન, મૌર્ય, સોલંકી, મૈત્રક, સિંધુ, હડપ્પા, etc.

RULES:
1. Fix garbled Gujarati words contextually (e.g., "સ્કેદગુપ" -> "સ્કંદગુપ્ત", "મોંહે-જો-દરો" -> "મોહેં-જો-દડો").
2. Fix random English characters/fragments that should be Gujarati words (e.g., "HAR" -> "મૂલર", "Casi" -> "બેટકા").
3. DO NOT change correctly recognized Gujarati words — preserve them exactly.
4. DO NOT change English words that are intentionally English (like exam names, proper nouns in English, technical terms).
5. PRESERVE question numbers exactly (001), 002), etc.).
6. PRESERVE option labels exactly: (A), (B), (C), (D).
7. PRESERVE exam references in parentheses exactly like (PI 38/2017-18), (STI 139/2020-21).
8. PRESERVE the exact line structure, line breaks, and formatting.
9. If a word is completely unrecoverable garbage, remove it rather than guessing wrong.
10. Output ONLY the corrected text. No explanations, no commentary, no markdown formatting.

RAW OCR TEXT:
"""


# GROQ VISION OCR (Multi-Key Rotation)

def _extract_retry_after_seconds(error_msg: str) -> int | None:
    """Best-effort extraction of retry-after seconds from API error text."""
    patterns = [
        r"retry[-_\s]?after[^0-9]*(\d+)",
        r"wait[^0-9]*(\d+)\s*(?:s|sec|seconds)",
    ]
    for pattern in patterns:
        match = re.search(pattern, error_msg, flags=re.IGNORECASE)
        if match:
            try:
                return max(1, int(match.group(1)))
            except (TypeError, ValueError):
                continue
    return None


def _classify_groq_error(error_msg: str) -> str:
    """Classify Groq error strings into retry/non-retry buckets."""
    msg = error_msg.lower()

    if "429" in msg or "rate_limit" in msg or ("rate" in msg and "limit" in msg):
        return "rate_limit"
    if "413" in msg or "too large" in msg or "request entity too large" in msg:
        return "payload_too_large"
    if "401" in msg or "403" in msg or "invalid api key" in msg or "unauthorized" in msg:
        return "auth_error"
    if re.search(r"\b5\d\d\b", msg) or "service unavailable" in msg or "internal server error" in msg:
        return "server_error"
    if "timeout" in msg or "timed out" in msg or "connection" in msg or "temporarily unavailable" in msg:
        return "network_error"
    return "fatal"


def should_skip_ai_correction(raw_text: str, min_nonspace_chars: int = AI_CORRECTION_MIN_NONSPACE_CHARS) -> bool:
    """Skip expensive AI correction for near-empty OCR output."""
    if not raw_text:
        return True
    meaningful_chars = len(re.sub(r"\s+", "", raw_text))
    return meaningful_chars < min_nonspace_chars

class GroqKeyPool:
    """
    Manages multiple Groq API keys with round-robin rotation and auto-fallback.
    When one key hits rate limits, automatically switches to the next key.
    
    Supports keys in .env as:
        GROQ_API_KEY_1=key1
        GROQ_API_KEY_2=key2
        GROQ_API_KEY_3=key3
    Or a single key:
        GROQ_API_KEY=key
    """

    def __init__(self, min_request_interval: float = GROQ_MIN_REQUEST_INTERVAL_SECONDS):
        self.available = False
        self.keys = self._load_keys()
        self.clients = []
        self.current_index = 0
        self.total_keys = len(self.keys)
        self.min_request_interval = max(0.0, float(min_request_interval))
        self.global_next_request_at = 0.0

        # Per-key adaptive state
        self.next_available_at = [0.0] * self.total_keys
        self.consecutive_429 = [0] * self.total_keys
        self.last_success_at = [0.0] * self.total_keys

        if not self.keys:
            print("⚠ No Groq API keys found. Continuing with raw OCR text fallback.")
            return

        try:
            from groq import Groq
            self.clients = [Groq(api_key=k) for k in self.keys]
            self.available = True
            print(f"Loaded {self.total_keys} Groq API key(s)")
        except Exception as e:
            print(f"⚠ Could not initialize Groq client ({str(e)[:120]}). Continuing without Groq correction.")
            self.clients = []
            self.available = False
            self.total_keys = 0
            self.next_available_at = []
            self.consecutive_429 = []
            self.last_success_at = []
    
    def _load_keys(self) -> list[str]:
        """Load all Groq API keys from Streamlit secrets, environment, or .env file."""
        keys = []
        env_vars = {}
        
        # 1. Check Streamlit secrets first (for Streamlit Cloud deployment)
        try:
            import streamlit as st
            for k, v in st.secrets.items():
                if k.startswith("GROQ_API_KEY") and v:
                    env_vars[k] = str(v)
        except Exception:
            pass  # Not running in Streamlit, skip
        
        # 2. Load from .env file (for local development)
        env_path = Path(__file__).parent / ".env"
        if env_path.exists():
            with open(env_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        k, v = line.split('=', 1)
                        key_name = k.strip()
                        key_val = v.strip().strip('"').strip("'")
                        if key_name not in env_vars:  # Don't override st.secrets
                            env_vars[key_name] = key_val
        
        # 3. Check actual environment variables
        for k, v in os.environ.items():
            if k.startswith("GROQ_API_KEY") and k not in env_vars:
                env_vars[k] = v
        
        # Collect numbered keys: GROQ_API_KEY_1, GROQ_API_KEY_2, ...
        numbered_keys = {}
        for k, v in env_vars.items():
            if k.startswith("GROQ_API_KEY_") and v and v != "your_groq_api_key_here":
                try:
                    num = int(k.split("_")[-1])
                    numbered_keys[num] = v
                except ValueError:
                    pass
        
        # Add numbered keys in order
        for num in sorted(numbered_keys.keys()):
            keys.append(numbered_keys[num])
        
        # If no numbered keys, try the single GROQ_API_KEY
        if not keys:
            single_key = env_vars.get("GROQ_API_KEY", "")
            if single_key and single_key != "your_groq_api_key_here":
                keys.append(single_key)
        
        return keys
    
    def get_client(self):
        """Get the current active Groq client (round-robin with fallback)."""
        now = time.time()
        
        # Try to find a non-exhausted key
        for _ in range(self.total_keys):
            if now >= self.exhausted_until[self.current_index]:
                return self.clients[self.current_index], self.current_index
            # This key is still rate-limited, try next
            self.current_index = (self.current_index + 1) % self.total_keys
        
        # All keys exhausted — wait for the one that recovers soonest
        soonest = min(self.exhausted_until)
        wait_time = max(0, soonest - now)
        if wait_time > 0:
            print(f"   ⏳ All {self.total_keys} keys rate-limited. Waiting {wait_time:.0f}s...")
            time.sleep(wait_time + 1)
        
        return self.clients[self.current_index], self.current_index
    
    def mark_rate_limited(self, key_index: int, cooldown: int = 60):
        """Mark a key as rate-limited for a cooldown period."""
        self.exhausted_until[key_index] = time.time() + cooldown
        key_num = key_index + 1
        print(f"Key #{key_num} rate-limited, cooling down {cooldown}s. Switching to next key...")
        # Rotate to next key
        self.current_index = (key_index + 1) % self.total_keys
    
    def wait_for_global_slot(self):
        """Apply global pacing between requests to avoid bursty 429 spikes."""
        if not self.available or self.min_request_interval <= 0:
            return

        now = time.time()
        if now < self.global_next_request_at:
            time.sleep(self.global_next_request_at - now)

        self.global_next_request_at = time.time() + self.min_request_interval

    def get_client(self, max_wait_seconds: float = GROQ_PAGE_MAX_WAIT_SECONDS):
        """
        Get the next available Groq client.
        If all keys are cooling down, wait until the earliest recovery (bounded by max_wait_seconds).
        Returns (client, key_index, waited_seconds) or (None, -1, waited_seconds).
        """
        if not self.available or self.total_keys == 0:
            return None, -1, 0.0

        waited = 0.0
        while waited <= max_wait_seconds:
            now = time.time()

            # Try to find a non-cooled-down key in round-robin order.
            for _ in range(self.total_keys):
                idx = self.current_index
                if now >= self.next_available_at[idx]:
                    return self.clients[idx], idx, waited
                self.current_index = (self.current_index + 1) % self.total_keys

            # All keys are cooling down; wait until earliest recovery.
            soonest = min(self.next_available_at)
            wait_time = max(0.0, soonest - now)
            if wait_time <= 0:
                continue

            if waited + wait_time > max_wait_seconds:
                return None, -1, waited

            print(f"   ⏳ All {self.total_keys} Groq keys cooling down. Waiting {wait_time:.1f}s...")
            sleep_for = min(wait_time + 0.05, max_wait_seconds - waited)
            time.sleep(sleep_for)
            waited += sleep_for

        return None, -1, waited

    def _compute_429_cooldown(self, key_index: int, error_msg: str = "") -> float:
        retry_after = _extract_retry_after_seconds(error_msg)
        if retry_after:
            base = float(retry_after)
        else:
            streak = max(1, self.consecutive_429[key_index])
            base = GROQ_BASE_429_COOLDOWN_SECONDS * (2 ** min(streak - 1, 5))
        jitter = random.uniform(0.4, 1.8)
        return min(float(GROQ_MAX_COOLDOWN_SECONDS), base + jitter)

    def mark_rate_limited(self, key_index: int, error_msg: str = "", cooldown: float | None = None) -> float:
        """Mark key as rate-limited with adaptive cooldown."""
        if key_index < 0 or key_index >= self.total_keys:
            return 0.0

        self.consecutive_429[key_index] += 1
        cooldown_seconds = float(cooldown) if cooldown is not None else self._compute_429_cooldown(key_index, error_msg)
        self.next_available_at[key_index] = time.time() + cooldown_seconds
        self.current_index = (key_index + 1) % self.total_keys
        return cooldown_seconds

    def mark_transient_error(self, key_index: int, cooldown: float = 8.0):
        """Temporarily cool a key on retryable server/network errors."""
        if key_index < 0 or key_index >= self.total_keys:
            return
        until = time.time() + max(0.5, float(cooldown))
        self.next_available_at[key_index] = max(self.next_available_at[key_index], until)
        self.current_index = (key_index + 1) % self.total_keys

    def mark_auth_error(self, key_index: int):
        """Sideline an unauthorized key for a long cooldown."""
        self.mark_transient_error(key_index, cooldown=3600.0)

    def mark_success(self, key_index: int):
        """Reset backoff state on success."""
        if key_index < 0 or key_index >= self.total_keys:
            return
        now = time.time()
        self.consecutive_429[key_index] = 0
        self.last_success_at[key_index] = now
        self.next_available_at[key_index] = min(self.next_available_at[key_index], now)

    def advance(self):
        """Move to the next key in round-robin (call after each successful request)."""
        if self.total_keys:
            self.current_index = (self.current_index + 1) % self.total_keys


def image_to_base64(image: Image.Image, max_size: int = 3800) -> str:
    """Convert PIL Image to base64 string, resizing if needed to stay under Groq limits."""
    # Resize if too large (Groq has 4MB base64 limit)
    width, height = image.size
    if width > max_size or height > max_size:
        ratio = min(max_size / width, max_size / height)
        new_size = (int(width * ratio), int(height * ratio))
        image = image.resize(new_size, Image.LANCZOS)
    
    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=90)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def ocr_page_groq(key_pool: GroqKeyPool, page_image: Image.Image, page_number: int, retry_count: int = 3) -> dict:
    """
    Use Groq Vision API to extract text from a page image.
    Uses the key pool for automatic rotation and fallback between multiple API keys.
    """
    img_b64 = image_to_base64(page_image)
    
    total_attempts = retry_count * key_pool.total_keys  # More attempts across all keys
    
    for attempt in range(total_attempts):
        client, key_idx = key_pool.get_client()
        key_num = key_idx + 1
        
        try:
            response = client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": GROQ_OCR_PROMPT
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{img_b64}"
                                }
                            }
                        ]
                    }
                ],
                temperature=0.0,
                max_completion_tokens=4096,
            )
            
            text = response.choices[0].message.content.strip()
            # Success — rotate to next key for next request (distribute load)
            key_pool.advance()
            return {
                "page_number": page_number,
                "text": text
            }
            
        except Exception as e:
            error_msg = str(e)
            if "429" in error_msg or "rate_limit" in error_msg.lower():
                # Rate limited — mark this key and auto-switch to next
                key_pool.mark_rate_limited(key_idx, cooldown=60)
                time.sleep(1)  # Brief pause before trying next key
            elif "413" in error_msg or "too large" in error_msg.lower():
                print(f"Image too large, reducing size...")
                img_b64 = image_to_base64(page_image, max_size=2500)
                time.sleep(2)
            else:
                if attempt < total_attempts - 1:
                    print(f"Error (key #{key_num}, page {page_number}): {error_msg[:100]}. Retrying...")
                    time.sleep(3)
                else:
                    print(f"Failed on page {page_number} after {total_attempts} attempts: {error_msg[:150]}")
                    return {
                        "page_number": page_number,
                        "text": ""
                    }
    
    return {"page_number": page_number, "text": ""}


def ocr_page_groq_robust(
    key_pool: GroqKeyPool,
    page_image: Image.Image,
    page_number: int,
    retry_count: int = 3,
    max_wait_seconds: int = GROQ_PAGE_MAX_WAIT_SECONDS,
) -> dict:
    """
    Reliability-first Groq vision OCR with adaptive per-key cooldown and bounded retries.
    """
    if not key_pool.available:
        return {"page_number": page_number, "text": ""}

    img_b64 = image_to_base64(page_image)
    total_attempts = min(
        GROQ_PAGE_MAX_ATTEMPTS,
        max(4, retry_count * max(1, key_pool.total_keys))
    )
    started_at = time.time()
    resized_for_limit = False
    attempts = 0

    while attempts < total_attempts and (time.time() - started_at) <= max_wait_seconds:
        remaining_wait = max(1.0, max_wait_seconds - (time.time() - started_at))
        client, key_idx, _ = key_pool.get_client(max_wait_seconds=remaining_wait)
        if client is None:
            print(f"[Page {page_number}] Groq wait ceiling reached. Returning empty OCR text.")
            break

        attempts += 1
        key_num = key_idx + 1
        key_pool.wait_for_global_slot()

        try:
            response = client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": GROQ_OCR_PROMPT},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}
                            }
                        ]
                    }
                ],
                temperature=0.0,
                max_completion_tokens=4096,
            )

            content = response.choices[0].message.content
            text = (content or "").strip()
            key_pool.mark_success(key_idx)
            key_pool.advance()
            return {"page_number": page_number, "text": text}

        except Exception as e:
            error_msg = str(e)
            category = _classify_groq_error(error_msg)

            if category == "rate_limit":
                cooldown = key_pool.mark_rate_limited(key_idx, error_msg=error_msg)
                print(
                    f"[Page {page_number}] 429 on key #{key_num}; cooldown {cooldown:.1f}s "
                    f"(attempt {attempts}/{total_attempts})."
                )
            elif category == "payload_too_large":
                if not resized_for_limit:
                    print(f"[Page {page_number}] Image too large, reducing size and retrying...")
                    img_b64 = image_to_base64(page_image, max_size=2500)
                    resized_for_limit = True
                key_pool.mark_transient_error(key_idx, cooldown=2.0)
            elif category in ("server_error", "network_error"):
                cooldown = min(20.0, 2.0 + attempts * 2.0)
                key_pool.mark_transient_error(key_idx, cooldown=cooldown)
                print(
                    f"[Page {page_number}] Transient Groq error on key #{key_num}; "
                    f"cooldown {cooldown:.1f}s, retrying..."
                )
            elif category == "auth_error":
                key_pool.mark_auth_error(key_idx)
                print(f"[Page {page_number}] Auth error on key #{key_num}. Key sidelined.")
            else:
                key_pool.mark_transient_error(key_idx, cooldown=1.0)
                print(f"[Page {page_number}] Non-retriable Groq error: {error_msg[:140]}")
                break

    print(f"[Page {page_number}] Groq OCR failed after {attempts} attempts.")
    return {"page_number": page_number, "text": ""}


# OPENROUTER VISION OCR (Multi-Key Pool + Model Cascade)

class OpenRouterKeyPool:
    """
    Manages multiple OpenRouter API keys with round-robin rotation + model cascade.

    Keys in .env as:
        OPENROUTER_API_KEY_1=key1
        OPENROUTER_API_KEY_2=key2
        ...
        OPENROUTER_API_KEY_20=key20
    Or a single key:
        OPENROUTER_API_KEY=key
    """

    def __init__(self):
        from openai import OpenAI
        self._OpenAI = OpenAI
        self.keys = self._load_keys()
        self.clients = [OpenAI(base_url=OPENROUTER_BASE_URL, api_key=k) for k in self.keys]
        self.current_index = 0
        self.total_keys = len(self.keys)

        # Track exhausted keys per model (key_index -> set of exhausted model names)
        self.exhausted_models = [{} for _ in range(self.total_keys)]  # key_idx -> {model: timestamp}

        # Current model index in the cascade
        self.current_model_index = 0

        print(f"✓ Loaded {self.total_keys} OpenRouter API key(s)")
        print(f"✓ Model cascade: {' → '.join(OPENROUTER_MODELS)}")

    def _load_keys(self) -> list[str]:
        """Load all OpenRouter API keys from Streamlit secrets, environment, or .env file."""
        keys = []
        env_vars = {}

        # 1. Check Streamlit secrets
        try:
            import streamlit as st
            for k, v in st.secrets.items():
                if k.startswith("OPENROUTER_API_KEY") and v:
                    env_vars[k] = str(v)
        except Exception:
            pass

        # 2. Load from .env file
        env_path = Path(__file__).parent / ".env"
        if env_path.exists():
            with open(env_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        k, v = line.split('=', 1)
                        key_name = k.strip()
                        key_val = v.strip().strip('"').strip("'")
                        if key_name.startswith("OPENROUTER_API_KEY") and key_name not in env_vars:
                            env_vars[key_name] = key_val

        # 3. Check environment variables
        for k, v in os.environ.items():
            if k.startswith("OPENROUTER_API_KEY") and k not in env_vars:
                env_vars[k] = v

        # Collect numbered keys: OPENROUTER_API_KEY_1, _2, ...
        numbered_keys = {}
        for k, v in env_vars.items():
            if k.startswith("OPENROUTER_API_KEY_") and v and v != "your_openrouter_api_key_here":
                try:
                    num = int(k.split("_")[-1])
                    numbered_keys[num] = v
                except ValueError:
                    pass

        # Add numbered keys in order
        for num in sorted(numbered_keys.keys()):
            keys.append(numbered_keys[num])

        # If no numbered keys, try single OPENROUTER_API_KEY
        if not keys:
            single_key = env_vars.get("OPENROUTER_API_KEY", "")
            if single_key and single_key != "your_openrouter_api_key_here":
                keys.append(single_key)

        if not keys:
            print("Error: No OpenRouter API keys found.")
            print("Add keys to .env file:")
            print("OPENROUTER_API_KEY_1=sk-or-v1-your_first_key")
            print("OPENROUTER_API_KEY_2=sk-or-v1-your_second_key")
            print("Get keys at: https://openrouter.ai/keys")
            sys.exit(1)

        return keys

    def get_client(self, model: str):
        """Get a non-exhausted client for the given model. Returns (client, key_index) or (None, -1)."""
        now = time.time()
        for _ in range(self.total_keys):
            idx = self.current_index
            exhausted_ts = self.exhausted_models[idx].get(model, 0)
            if now >= exhausted_ts:
                return self.clients[idx], idx
            self.current_index = (self.current_index + 1) % self.total_keys

        # All keys exhausted for this model
        return None, -1

    def mark_exhausted(self, key_index: int, model: str, cooldown: int = 300):
        """Mark a key as exhausted for a specific model (402 = long cooldown, 429 = short)."""
        self.exhausted_models[key_index][model] = time.time() + cooldown
        key_num = key_index + 1
        print(f"   Key #{key_num} exhausted for {model.split('/')[-1]}, cooldown {cooldown}s")
        self.current_index = (key_index + 1) % self.total_keys

    def advance(self):
        """Move to next key (round-robin after success)."""
        self.current_index = (self.current_index + 1) % self.total_keys


def ocr_page_openrouter(key_pool: OpenRouterKeyPool, page_image: Image.Image, page_number: int) -> dict:
    """
    Use OpenRouter Vision API with model cascade + multi-key rotation.
    Tries: qwen-7b → qwen-72b → gemma-3-27b:free
    Rotates through all API keys before falling to next model.
    """
    img_b64 = image_to_base64(page_image)

    for model_idx, model in enumerate(OPENROUTER_MODELS):
        is_free = ":free" in model
        max_attempts = 8 if is_free else 4

        for attempt in range(max_attempts):
            client, key_idx = key_pool.get_client(model)

            if client is None:
                # All keys exhausted for this model — fall to next model
                print(f"   All keys exhausted for {model.split('/')[-1]}. Trying next model...")
                break

            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": GROQ_OCR_PROMPT},
                                {
                                    "type": "image_url",
                                    "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}
                                }
                            ]
                        }
                    ],
                    temperature=0.0,
                    max_tokens=4096,
                    extra_headers={
                        "HTTP-Referer": "https://github.com/DocuMorph",
                        "X-Title": "DocuMorph"
                    }
                )

                content = response.choices[0].message.content
                if content is None:
                    raise Exception("API returned no content (blocked/filtered)")

                text = content.strip()
                key_pool.advance()
                return {"page_number": page_number, "text": text}

            except Exception as e:
                error_msg = str(e)

                if "404" in error_msg or "No endpoints" in error_msg:
                    # Model doesn't exist — skip to next model immediately
                    print(f"   Model {model.split('/')[-1]} not available. Skipping to next...")
                    break

                elif "402" in error_msg or "Payment Required" in error_msg:
                    # No credits on this key for this model — mark exhausted permanently
                    key_pool.mark_exhausted(key_idx, model, cooldown=99999)

                elif "429" in error_msg or "rate" in error_msg.lower():
                    if is_free:
                        wait = min(5 * (attempt + 1), 30)
                        print(f"   ⏳ Rate limited [free] (attempt {attempt + 1}/{max_attempts}). Waiting {wait}s...")
                        time.sleep(wait)
                    else:
                        # Paid model rate limited — try next key
                        key_pool.mark_exhausted(key_idx, model, cooldown=30)

                elif "413" in error_msg or "too large" in error_msg.lower():
                    img_b64 = image_to_base64(page_image, max_size=2500)
                    time.sleep(1)

                elif "onnection" in error_msg:
                    wait = 3 * (attempt + 1)
                    print(f"   Connection error (attempt {attempt + 1}). Waiting {wait}s...")
                    time.sleep(wait)

                elif attempt < max_attempts - 1:
                    print(f"   Error [{model.split('/')[-1]}] (attempt {attempt + 1}): {error_msg[:100]}. Retrying...")
                    time.sleep(2)
                else:
                    break  # Try next model

    print(f"   ✗ Failed on page {page_number} — all models and keys exhausted")
    return {"page_number": page_number, "text": ""}


def extract_text_openrouter(pdf_path: str, start_page: int = None, end_page: int = None) -> list[dict]:
    """
    Extract text from PDF pages using OpenRouter Vision API.
    Uses multi-key pool with model cascade (qwen-7b → qwen-72b → gemma-free).
    """
    key_pool = OpenRouterKeyPool()

    # Determine page range
    if start_page and end_page:
        first_page = start_page
        last_page = end_page
    else:
        total_pages = _get_page_count(pdf_path)
        first_page = start_page or 1
        last_page = end_page or total_pages
        if last_page == 0:
            print("Could not detect page count, trying page 1...")
            last_page = first_page

    total = last_page - first_page + 1
    print(f"Pages to process: {first_page} to {last_page} ({total} pages)")
    print(f"---")

    pages_text = []

    for page_num in range(first_page, last_page + 1):
        i = page_num - first_page
        print(f"[Page {page_num}] Converting to image (DPI=150)...")

        try:
            img = _convert_single_page(pdf_path, page_num, dpi=150)
        except Exception as e:
            print(f"[Page {page_num}] ERROR converting: {str(e)[:150]}")
            pages_text.append({"page_number": page_num, "text": ""})
            continue

        if img is None:
            print(f"[Page {page_num}] WARNING: No image returned, skipping.")
            pages_text.append({"page_number": page_num, "text": ""})
            continue

        w, h = img.size
        print(f"[Page {page_num}] Image: {w}x{h}px. Sending to OpenRouter...")

        result = ocr_page_openrouter(key_pool, img, page_num)
        pages_text.append(result)

        text_len = len(result["text"])
        print(f"[Page {page_num}] Done. Extracted {text_len} chars. ({i + 1}/{total})")

        # Free memory aggressively
        del img
        gc.collect()

        # Brief cooldown between requests (key rotation handles rate limits)
        if i < total - 1:
            time.sleep(0.3)

        # Periodic save (checkpoint) every 10 pages in case of crash/memory issue
        if len(pages_text) % 10 == 0:
            checkpoint_path = f"{Path(pdf_path).stem}_checkpoint.json"
            with open(checkpoint_path, 'w', encoding='utf-8') as f:
                json.dump({"processed_pages": len(pages_text), "pages": pages_text}, f, ensure_ascii=False)
            print(f"   [Checkpoint saved: {checkpoint_path}]")

    print(f"---")
    print(f"OCR complete. Processed {len(pages_text)} pages.")
    return pages_text


def _get_page_count(pdf_path: str) -> int:
    """Get total page count of a PDF using PyMuPDF (fast)."""
    try:
        with fitz.open(pdf_path) as doc:
            return len(doc)
    except Exception as e:
        print(f"Error getting page count: {e}")
        return 0


def _convert_single_page(pdf_path: str, page_num: int, dpi: int = 150) -> Image.Image:
    """Convert a single PDF page to an image using PyMuPDF. Very fast and memory-efficient."""
    # page_num is 1-indexed, PyMuPDF is 0-indexed
    try:
        with fitz.open(pdf_path) as doc:
            page = doc.load_page(page_num - 1)
            # Zoom matrix for DPI. 72 is default PDF resolution
            zoom = dpi / 72.0
            mat = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=mat, alpha=False)
            
            # Convert PyMuPDF pixmap to PIL Image
            mode = "RGB" if pix.n >= 3 else "L"
            img = Image.frombytes(mode, [pix.width, pix.height], pix.samples)
            
            # Explicitly free PyMuPDF objects
            pix = None
            page = None
            
            return img
    except Exception as e:
        print(f"Error converting page {page_num}: {e}")
        return None


def extract_text_groq(pdf_path: str, start_page: int = None, end_page: int = None) -> list[dict]:
    """
    Extract text from PDF pages using Groq Vision API.
    Processes pages ONE AT A TIME to avoid memory issues.
    Uses multiple API keys with round-robin rotation.
    """
    key_pool = GroqKeyPool()
    
    # Determine page range
    if start_page and end_page:
        first_page = start_page
        last_page = end_page
    else:
        total_pages = _get_page_count(pdf_path)
        first_page = start_page or 1
        last_page = end_page or total_pages
        if last_page == 0:
            # Fallback: try converting and see
            print("Could not detect page count, trying page 1...")
            last_page = first_page
    
    total = last_page - first_page + 1
    print(f"Pages to process: {first_page} to {last_page} ({total} pages)")
    print(f"OCR Engine: Groq Vision (Llama 4 Scout)")
    print(f"API Keys loaded: {key_pool.total_keys}")
    print(f"---")
    
    pages_text = []
    
    for page_num in range(first_page, last_page + 1):
        i = page_num - first_page
        print(f"[Page {page_num}] Converting to image (DPI=200)...")
        
        try:
            img = _convert_single_page(pdf_path, page_num, dpi=200)
        except Exception as e:
            print(f"[Page {page_num}] ERROR converting: {str(e)[:150]}")
            pages_text.append({"page_number": page_num, "text": ""})
            continue
        
        if img is None:
            print(f"[Page {page_num}] WARNING: No image returned, skipping.")
            pages_text.append({"page_number": page_num, "text": ""})
            continue
        
        w, h = img.size
        print(f"[Page {page_num}] Image: {w}x{h}px. Sending to Groq...")
        
        result = ocr_page_groq_robust(key_pool, img, page_num)
        pages_text.append(result)
        
        text_len = len(result["text"])
        print(f"[Page {page_num}] Done. Extracted {text_len} chars. ({i + 1}/{total})")
        
        # Free memory
        del img
        
        # Small delay between requests
        if i < total - 1:
            time.sleep(1)
    
    print(f"---")
    print(f"OCR complete. Processed {len(pages_text)} pages.")
    return pages_text


# TESSERACT OCR (FALLBACK) 

def preprocess_image(image: Image.Image) -> Image.Image:
    """
    Preprocess PDF page image for better OCR accuracy.
    - Convert to grayscale
    - Increase contrast
    - Sharpen
    - Upscale for better character recognition
    """
    # Convert to grayscale
    img = image.convert('L')
    
    # Upscale 2x for better OCR on small text
    width, height = img.size
    img = img.resize((width * 2, height * 2), Image.LANCZOS)
    
    # Increase contrast
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(2.0)
    
    # Sharpen
    img = img.filter(ImageFilter.SHARPEN)
    
    # Binarize (threshold)
    img = img.point(lambda x: 0 if x < 140 else 255, '1')
    
    return img


def ocr_image_tesseract(image: Image.Image) -> str:
    """Run Tesseract OCR on a preprocessed image and return text."""
    import pytesseract
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH
    
    processed = preprocess_image(image)
    
    # PSM 6 = Assume a single uniform block of text
    custom_config = r'--oem 3 --psm 6'
    text = pytesseract.image_to_string(
        processed,
        lang='guj+eng',
        config=custom_config
    )
    return text


def ocr_page_two_columns(page_image: Image.Image, page_number: int) -> dict:
    """
    Split a two-column PDF page into left and right halves,
    OCR each separately with Tesseract, and concatenate left then right.
    """
    width, height = page_image.size
    
    # Small overlap margin to avoid cutting through text (2% of width)
    margin = int(width * 0.02)
    
    # Crop left column (left half)
    left_col = page_image.crop((0, 0, width // 2 + margin, height))
    
    # Crop right column (right half)
    right_col = page_image.crop((width // 2 - margin, 0, width, height))
    
    # OCR each column separately
    left_text = ocr_image_tesseract(left_col)
    right_text = ocr_image_tesseract(right_col)
    
    # Combine: left column first, then right column
    combined_text = left_text.strip() + "\n\n" + right_text.strip()
    
    return {
        "page_number": page_number,
        "text": combined_text
    }


def extract_text_tesseract(pdf_path: str, start_page: int = None, end_page: int = None) -> list[dict]:
    """
    Convert PDF pages to images and run Tesseract OCR on each.
    Processes pages ONE AT A TIME. Uses two-column splitting.
    """
    # Determine page range
    if start_page and end_page:
        first_page = start_page
        last_page = end_page
    else:
        total_pages = _get_page_count(pdf_path)
        first_page = start_page or 1
        last_page = end_page or total_pages
        if last_page == 0:
            last_page = first_page
    
    total = last_page - first_page + 1
    print(f"Pages to process: {first_page} to {last_page} ({total} pages)")
    print(f"OCR Engine: Tesseract (Gujarati + English)")
    print(f"---")
    
    pages_text = []
    
    for page_num in range(first_page, last_page + 1):
        i = page_num - first_page
        print(f"[Page {page_num}] Converting to image (DPI=300)...")
        
        try:
            img = _convert_single_page(pdf_path, page_num, dpi=300)
        except Exception as e:
            print(f"[Page {page_num}] ERROR converting: {str(e)[:150]}")
            pages_text.append({"page_number": page_num, "text": ""})
            continue
        
        if img is None:
            print(f"[Page {page_num}] WARNING: No image returned, skipping.")
            pages_text.append({"page_number": page_num, "text": ""})
            continue
        
        print(f"[Page {page_num}] Running Tesseract 2-column OCR...")
        result = ocr_page_two_columns(img, page_num)
        pages_text.append(result)
        
        text_len = len(result["text"])
        print(f"[Page {page_num}] Done. Extracted {text_len} chars. ({i + 1}/{total})")
        
        # Free memory
        del img
    
    print(f"---")
    print(f"OCR complete. Processed {len(pages_text)} pages.")
    return pages_text


def extract_text_tesseract_groq_dual(pdf_path: str, start_page: int = None, end_page: int = None) -> tuple[list[dict], list[dict]]:
    """
    HYBRID DUAL ENGINE: Tesseract OCR + Groq text correction.

    Returns:
        (raw_pages_text, fixed_pages_text)
    """
    groq_pool = GroqKeyPool()

    # Determine page range
    if start_page and end_page:
        first_page = start_page
        last_page = end_page
    else:
        total_pages = _get_page_count(pdf_path)
        first_page = start_page or 1
        last_page = end_page or total_pages
        if last_page == 0:
            last_page = first_page

    total = last_page - first_page + 1
    print(f"Pages to process: {first_page} to {last_page} ({total} pages)")
    print(f"OCR Engine: Tesseract + Groq Gujarati Correction")
    print(f"---")

    raw_pages_text = []
    fixed_pages_text = []
    groq_fallback_used = False

    for page_num in range(first_page, last_page + 1):
        i = page_num - first_page
        print(f"[Page {page_num}] Step 1/2: Tesseract OCR (DPI=300)...")

        try:
            img = _convert_single_page(pdf_path, page_num, dpi=300)
        except Exception as e:
            print(f"[Page {page_num}] ERROR converting: {str(e)[:150]}")
            raw_pages_text.append({"page_number": page_num, "text": ""})
            fixed_pages_text.append({"page_number": page_num, "text": ""})
            continue

        if img is None:
            print(f"[Page {page_num}] WARNING: No image returned, skipping.")
            raw_pages_text.append({"page_number": page_num, "text": ""})
            fixed_pages_text.append({"page_number": page_num, "text": ""})
            continue

        raw_result = ocr_page_two_columns(img, page_num)
        raw_text = raw_result.get("text", "")
        raw_pages_text.append(raw_result)
        raw_len = len(raw_text)
        print(f"[Page {page_num}] Tesseract extracted {raw_len} chars.")

        del img
        gc.collect()

        fixed_text = raw_text
        if should_skip_ai_correction(raw_text):
            print(f"[Page {page_num}] Step 2/2: Skipping Groq fix (near-empty text).")
        elif not groq_pool.available:
            if not groq_fallback_used:
                print("⚠ Groq unavailable. Continuing with raw Tesseract text.")
                groq_fallback_used = True
        else:
            print(f"[Page {page_num}] Step 2/2: Groq correcting Gujarati text...")
            fixed_text = fix_text_with_ai_groq_robust(groq_pool, raw_text, page_num)

        fixed_pages_text.append({"page_number": page_num, "text": fixed_text})
        fixed_len = len(fixed_text)
        print(f"[Page {page_num}] Done. Raw→Fixed chars: {raw_len} → {fixed_len}. ({i + 1}/{total})")

        if i < total - 1:
            time.sleep(0.3)

        # Checkpoint every 10 pages
        if len(raw_pages_text) % 10 == 0:
            checkpoint_path = f"{Path(pdf_path).stem}_tesseract_groq_checkpoint.json"
            with open(checkpoint_path, 'w', encoding='utf-8') as f:
                json.dump(
                    {
                        "processed_pages": len(raw_pages_text),
                        "raw_pages": raw_pages_text,
                        "fixed_pages": fixed_pages_text
                    },
                    f,
                    ensure_ascii=False
                )
            print(f"   [Checkpoint saved: {checkpoint_path}]")

    print(f"---")
    print(f"Hybrid OCR complete. Processed {len(raw_pages_text)} pages.")
    return raw_pages_text, fixed_pages_text


# AI TEXT CORRECTION (for fixing Tesseract OCR errors)

def fix_text_with_ai_groq(key_pool: GroqKeyPool, raw_text: str, page_number: int, retry_count: int = 3) -> str:
    """
    Send raw Tesseract OCR text to Groq (text-only, no images) to fix garbled words.
    Much cheaper than vision OCR since we only send text.
    """
    if not raw_text.strip():
        return raw_text

    total_attempts = retry_count * key_pool.total_keys

    for attempt in range(total_attempts):
        client, key_idx = key_pool.get_client()
        key_num = key_idx + 1

        try:
            response = client.chat.completions.create(
                model=GROQ_TEXT_MODEL,
                messages=[
                    {
                        "role": "user",
                        "content": AI_TEXT_FIX_PROMPT + raw_text
                    }
                ],
                temperature=0.0,
                max_completion_tokens=4096,
            )

            fixed_text = response.choices[0].message.content.strip()
            key_pool.advance()
            return fixed_text

        except Exception as e:
            error_msg = str(e)
            if "429" in error_msg or "rate_limit" in error_msg.lower():
                key_pool.mark_rate_limited(key_idx, cooldown=60)
                time.sleep(1)
            elif attempt < total_attempts - 1:
                print(f"   AI fix error (key #{key_num}, page {page_number}): {error_msg[:100]}. Retrying...")
                time.sleep(3)
            else:
                print(f"   AI fix failed on page {page_number}, using raw text. Error: {error_msg[:150]}")
                return raw_text  # Fallback to raw Tesseract text

    return raw_text  # Fallback


def fix_text_with_ai_groq_robust(
    key_pool: GroqKeyPool,
    raw_text: str,
    page_number: int,
    retry_count: int = 3,
    max_wait_seconds: int = GROQ_TEXT_MAX_WAIT_SECONDS,
) -> str:
    """
    Reliability-first Groq text correction with adaptive cooldown and bounded retries.
    """
    if not raw_text.strip() or should_skip_ai_correction(raw_text):
        return raw_text

    if not key_pool.available:
        return raw_text

    total_attempts = min(
        GROQ_TEXT_MAX_ATTEMPTS,
        max(3, retry_count * max(1, key_pool.total_keys))
    )
    started_at = time.time()
    attempts = 0

    while attempts < total_attempts and (time.time() - started_at) <= max_wait_seconds:
        remaining_wait = max(1.0, max_wait_seconds - (time.time() - started_at))
        client, key_idx, _ = key_pool.get_client(max_wait_seconds=remaining_wait)
        if client is None:
            print(f"[Page {page_number}] Groq correction wait ceiling reached. Using raw text.")
            break

        attempts += 1
        key_num = key_idx + 1
        key_pool.wait_for_global_slot()

        try:
            response = client.chat.completions.create(
                model=GROQ_TEXT_MODEL,
                messages=[
                    {
                        "role": "user",
                        "content": AI_TEXT_FIX_PROMPT + raw_text
                    }
                ],
                temperature=0.0,
                max_completion_tokens=4096,
            )

            content = response.choices[0].message.content
            fixed_text = (content or "").strip()
            key_pool.mark_success(key_idx)
            key_pool.advance()
            return fixed_text if fixed_text else raw_text

        except Exception as e:
            error_msg = str(e)
            category = _classify_groq_error(error_msg)

            if category == "rate_limit":
                cooldown = key_pool.mark_rate_limited(key_idx, error_msg=error_msg)
                print(
                    f"[Page {page_number}] 429 during Groq correction on key #{key_num}; "
                    f"cooldown {cooldown:.1f}s (attempt {attempts}/{total_attempts})."
                )
            elif category in ("server_error", "network_error"):
                cooldown = min(20.0, 2.0 + attempts * 2.0)
                key_pool.mark_transient_error(key_idx, cooldown=cooldown)
                print(
                    f"[Page {page_number}] Transient Groq correction error on key #{key_num}; "
                    f"cooldown {cooldown:.1f}s."
                )
            elif category == "auth_error":
                key_pool.mark_auth_error(key_idx)
                print(f"[Page {page_number}] Groq auth error on key #{key_num}. Key sidelined.")
            else:
                key_pool.mark_transient_error(key_idx, cooldown=1.0)
                print(f"[Page {page_number}] Groq correction failed: {error_msg[:140]}")
                break

    print(f"[Page {page_number}] Groq correction failed after {attempts} attempts. Using raw text.")
    return raw_text


def fix_text_with_ai_openrouter(key_pool: OpenRouterKeyPool, raw_text: str, page_number: int) -> str:
    """
    Send raw Tesseract OCR text to OpenRouter (text-only) to fix garbled words.
    Uses text-only models (much cheaper than vision models).
    """
    if not raw_text.strip():
        return raw_text

    for model_idx, model in enumerate(OPENROUTER_TEXT_MODELS):
        is_free = ":free" in model
        max_attempts = 8 if is_free else 4

        for attempt in range(max_attempts):
            client, key_idx = key_pool.get_client(model)

            if client is None:
                print(f"   All keys exhausted for {model.split('/')[-1]}. Trying next model...")
                break

            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {
                            "role": "user",
                            "content": AI_TEXT_FIX_PROMPT + raw_text
                        }
                    ],
                    temperature=0.0,
                    max_tokens=4096,
                    extra_headers={
                        "HTTP-Referer": "https://github.com/DocuMorph",
                        "X-Title": "DocuMorph"
                    }
                )

                content = response.choices[0].message.content
                if content is None:
                    raise Exception("API returned no content")

                fixed_text = content.strip()
                key_pool.advance()
                return fixed_text

            except Exception as e:
                error_msg = str(e)

                if "404" in error_msg or "No endpoints" in error_msg:
                    print(f"   Model {model.split('/')[-1]} not available. Trying next...")
                    break

                elif "402" in error_msg or "Payment Required" in error_msg:
                    key_pool.mark_exhausted(key_idx, model, cooldown=99999)

                elif "429" in error_msg or "rate" in error_msg.lower():
                    if is_free:
                        wait = min(5 * (attempt + 1), 30)
                        print(f"   ⏳ Rate limited [free] (attempt {attempt + 1}/{max_attempts}). Waiting {wait}s...")
                        time.sleep(wait)
                    else:
                        key_pool.mark_exhausted(key_idx, model, cooldown=30)

                elif attempt < max_attempts - 1:
                    print(f"   AI fix error [{model.split('/')[-1]}] (attempt {attempt + 1}): {error_msg[:100]}. Retrying...")
                    time.sleep(2)
                else:
                    break

    print(f"   AI fix failed on page {page_number}, using raw text.")
    return raw_text  # Fallback


def extract_text_tesseract_ai(pdf_path: str, start_page: int = None, end_page: int = None,
                               ai_provider: str = "openrouter") -> list[dict]:
    """
    HYBRID ENGINE: Tesseract OCR + AI Text Correction.

    Step 1: Extract raw text with Tesseract (fast, free, local)
    Step 2: Send raw text to AI model to fix garbled/broken words
    Step 3: Return corrected text

    This is MUCH CHEAPER than vision OCR because we send text (~2KB)
    instead of images (~3MB) to the API.
    """
    # Initialize AI key pool
    if ai_provider == "groq":
        groq_pool = GroqKeyPool()
        openrouter_pool = None
    else:
        groq_pool = None
        openrouter_pool = OpenRouterKeyPool()

    # Try to initialize both for fallback
    try:
        if groq_pool is None:
            groq_pool = GroqKeyPool()
    except SystemExit:
        groq_pool = None
    try:
        if openrouter_pool is None:
            openrouter_pool = OpenRouterKeyPool()
    except SystemExit:
        openrouter_pool = None

    # Determine page range
    if start_page and end_page:
        first_page = start_page
        last_page = end_page
    else:
        total_pages = _get_page_count(pdf_path)
        first_page = start_page or 1
        last_page = end_page or total_pages
        if last_page == 0:
            last_page = first_page

    total = last_page - first_page + 1
    print(f"Pages to process: {first_page} to {last_page} ({total} pages)")
    print(f"OCR Engine: Tesseract + AI Text Correction")
    print(f"Step 1: Tesseract (local) → Step 2: AI fix ({ai_provider})")
    print(f"---")

    pages_text = []

    for page_num in range(first_page, last_page + 1):
        i = page_num - first_page
        print(f"[Page {page_num}] Step 1: Tesseract OCR (DPI=300)...")

        try:
            img = _convert_single_page(pdf_path, page_num, dpi=300)
        except Exception as e:
            print(f"[Page {page_num}] ERROR converting: {str(e)[:150]}")
            pages_text.append({"page_number": page_num, "text": ""})
            continue

        if img is None:
            print(f"[Page {page_num}] WARNING: No image returned, skipping.")
            pages_text.append({"page_number": page_num, "text": ""})
            continue

        # Step 1: Tesseract OCR (two-column)
        raw_result = ocr_page_two_columns(img, page_num)
        raw_text = raw_result["text"]
        raw_len = len(raw_text)
        print(f"[Page {page_num}] Tesseract extracted {raw_len} chars.")

        # Free image memory
        del img
        gc.collect()

        # Step 2: AI text correction
        if raw_text.strip():
            print(f"[Page {page_num}] Step 2: AI fixing garbled words...")

            fixed_text = None

            # Try primary provider first
            if ai_provider == "groq" and groq_pool:
                fixed_text = fix_text_with_ai_groq_robust(groq_pool, raw_text, page_num)
            elif ai_provider == "openrouter" and openrouter_pool:
                fixed_text = fix_text_with_ai_openrouter(openrouter_pool, raw_text, page_num)

            # If primary failed or returned raw text, try fallback
            if fixed_text is None or fixed_text == raw_text:
                if ai_provider == "groq" and openrouter_pool:
                    print(f"[Page {page_num}] Groq failed, trying OpenRouter fallback...")
                    fixed_text = fix_text_with_ai_openrouter(openrouter_pool, raw_text, page_num)
                elif ai_provider == "openrouter" and groq_pool:
                    print(f"[Page {page_num}] OpenRouter failed, trying Groq fallback...")
                    fixed_text = fix_text_with_ai_groq_robust(groq_pool, raw_text, page_num)

            if fixed_text is None:
                fixed_text = raw_text

            fixed_len = len(fixed_text)
            print(f"[Page {page_num}] AI corrected: {raw_len} → {fixed_len} chars. ({i + 1}/{total})")
            pages_text.append({"page_number": page_num, "text": fixed_text})
        else:
            print(f"[Page {page_num}] Empty text, skipping AI fix. ({i + 1}/{total})")
            pages_text.append({"page_number": page_num, "text": ""})

        # Brief cooldown
        if i < total - 1:
            time.sleep(0.5)

        # Checkpoint every 10 pages
        if len(pages_text) % 10 == 0:
            checkpoint_path = f"{Path(pdf_path).stem}_checkpoint.json"
            with open(checkpoint_path, 'w', encoding='utf-8') as f:
                json.dump({"processed_pages": len(pages_text), "pages": pages_text}, f, ensure_ascii=False)
            print(f"   [Checkpoint saved: {checkpoint_path}]")

    print(f"---")
    print(f"Hybrid OCR complete. Processed {len(pages_text)} pages.")
    return pages_text


def enhance_pages_with_ai(pages_text: list[dict], ai_provider: str = "openrouter") -> list[dict]:
    """
    Takes an existing list of raw extracted pages (from Tesseract) and applies
    AI text correction to each page individually. Returns a new list of corrected pages.
    """
    print(f"---")
    print(f"Starting AI Enhancement on {len(pages_text)} pages...")
    
    # Initialize AI key pool
    if ai_provider == "groq":
        groq_pool = GroqKeyPool()
        openrouter_pool = None
    else:
        groq_pool = None
        openrouter_pool = OpenRouterKeyPool()

    # Try to initialize both for fallback
    try:
        if groq_pool is None:
            groq_pool = GroqKeyPool()
    except SystemExit:
        groq_pool = None
    try:
        if openrouter_pool is None:
            openrouter_pool = OpenRouterKeyPool()
    except SystemExit:
        openrouter_pool = None

    enhanced_pages = []
    total = len(pages_text)

    for i, page_data in enumerate(pages_text):
        page_num = page_data.get("page_number", i + 1)
        raw_text = page_data.get("text", "")
        raw_len = len(raw_text)

        if not raw_text.strip():
            print(f"[Page {page_num}] Empty text, skipping AI fix. ({i + 1}/{total})")
            enhanced_pages.append({"page_number": page_num, "text": ""})
            continue

        print(f"[Page {page_num}] AI fixing garbled words...")
        fixed_text = None

        # Try primary provider first
        if ai_provider == "groq" and groq_pool:
            fixed_text = fix_text_with_ai_groq_robust(groq_pool, raw_text, page_num)
        elif ai_provider == "openrouter" and openrouter_pool:
            fixed_text = fix_text_with_ai_openrouter(openrouter_pool, raw_text, page_num)

        # If primary failed or returned raw text, try fallback
        if fixed_text is None or fixed_text == raw_text:
            if ai_provider == "groq" and openrouter_pool:
                print(f"[Page {page_num}] Groq failed, trying OpenRouter fallback...")
                fixed_text = fix_text_with_ai_openrouter(openrouter_pool, raw_text, page_num)
            elif ai_provider == "openrouter" and groq_pool:
                print(f"[Page {page_num}] OpenRouter failed, trying Groq fallback...")
                fixed_text = fix_text_with_ai_groq_robust(groq_pool, raw_text, page_num)

        if fixed_text is None:
            fixed_text = raw_text

        fixed_len = len(fixed_text)
        print(f"[Page {page_num}] AI corrected: {raw_len} → {fixed_len} chars. ({i + 1}/{total})")
        
        enhanced_pages.append({"page_number": page_num, "text": fixed_text})
        
        # Brief cooldown
        if i < total - 1:
            time.sleep(0.5)

    print(f"AI Enhancement complete.")
    return enhanced_pages


# QUESTION PARSING 

def parse_questions(pages_text: list[dict]) -> list[dict]:
    """
    Parse all questions from OCR-extracted text.
    
    Handles the Gujarati MCQ format:
        001) Question text here? (EXAM_REF)
            (A) Option A    (B) Option B
            (C) Option C    (D) Option D
    """
    # Combine all text with page markers
    full_text_segments = []
    for page_data in pages_text:
        full_text_segments.append(f"\n<<<PAGE_{page_data['page_number']}>>>\n")
        full_text_segments.append(page_data["text"])
    
    full_text = "\n".join(full_text_segments)
    
    # Clean answer keys like "1-C /2-D/3-A/4-A/5-D/6-A" before parsing
    full_text = re.sub(r'\d+-[A-D]\s*/\s*\d+-[A-D](?:\s*/\s*\d+-[A-D])*(?:\s*\|?\s*)?', '', full_text)
    
    # Pattern to match question numbers like "001)", "002)", "1234)", etc.
    question_pattern = re.compile(
        r'(?:^|\n)\s*(\d{1,4})\s*\)\s*(.+?)(?=(?:\n\s*\d{1,4}\s*\)\s)|<<<PAGE_|\Z)',
        re.DOTALL
    )
    
    matches = list(question_pattern.finditer(full_text))
    print(f"\nFound {len(matches)} question blocks")
    
    questions = []
    
    for idx, match in enumerate(matches):
        q_num = match.group(1).strip()
        q_block = match.group(2).strip()
        
        # Determine which page this question is on
        page_number = 1
        page_markers = list(re.finditer(r'<<<PAGE_(\d+)>>>', full_text[:match.start()]))
        if page_markers:
            page_number = int(page_markers[-1].group(1))
        
        # Extract exam reference
        exam_ref = extract_exam_reference(q_block)
        
        # Extract options (A), (B), (C), (D)
        options = extract_options(q_block)
        
        # Extract question text (everything before the first option)
        question_text = extract_question_text(q_block)
        
        # Clean up
        question_text = clean_text(question_text)
        
        question_data = {
            "id": len(questions) + 1,
            "question_number": q_num.zfill(3),
            "question_text": question_text,
            "options": options,
            "page_number": page_number
        }
        
        if exam_ref:
            question_data["exam_reference"] = exam_ref
        
        questions.append(question_data)
        
        if len(questions) % 500 == 0:
            print(f"   Parsed {len(questions)} questions...")
    
    return questions


def extract_exam_reference(q_block: str) -> str | None:
    """Extract exam reference like (PI 38/2017-18), (STI 139/2020-21), etc."""
    patterns = [
        re.compile(r'\(([A-Z][A-Za-z]*(?:\s+[A-Za-z]+)*\s*[\-]?\s*\d+\s*/\s*\d{4}(?:\s*-\s*\d{2,4})?)\)'),
        re.compile(r'\(([^()]*\d+\s*/\s*\d{4}(?:\s*-\s*\d{2,4})?)\)'),
    ]
    
    for pattern in patterns:
        match = pattern.search(q_block)
        if match:
            candidate = match.group(1).strip()
            # Make sure it's not an option label
            if not re.match(r'^[A-D]$', candidate) and len(candidate) > 3:
                return candidate
    
    return None


def normalize_option_labels(text: str) -> str:
    """
    Fix common OCR misreadings of option labels.
    Tesseract often confuses:
        (B) → (3) or (8)
        (C) → (0) or (૦) 
        (D) → (2) or (12)
    """
    # (8) → (B) — very common OCR confusion
    text = re.sub(r'\(\s*8\s*\)\s', '(B) ', text)
    # (3) → (B) — when appearing after (A) context  
    text = re.sub(r'\(\s*3\s*\)\s', '(B) ', text)
    # (૦) → (C) — Gujarati zero confused with C
    text = re.sub(r'\(\s*૦\s*\)\s', '(C) ', text)
    # (0) → (C) — zero confused with C
    text = re.sub(r'\(\s*0\s*\)\s', '(C) ', text)
    # (€) → (C) — Euro symbol confused with C
    text = re.sub(r'\(\s*€\s*\)\s*', '(C) ', text)
    # (12) → (D)
    text = re.sub(r'\(\s*12\s*\)\s', '(D) ', text)
    # (2) → (D) — common confusion
    text = re.sub(r'\(\s*2\s*\)\s', '(D) ', text)
    
    return text


def extract_options(q_block: str) -> dict:
    """
    Extract options (A), (B), (C), (D) from a question block.
    Handles OCR-introduced spacing, line break issues, and label misreadings.
    """
    options = {}
    
    # First normalize OCR-misread option labels
    normalized_block = normalize_option_labels(q_block)
    
    # Collect lines containing option markers
    lines = normalized_block.split('\n')
    option_text = ""
    in_options = False
    
    for line in lines:
        # Look for option markers
        if re.search(r'\(\s*[A-D]\s*\)', line):
            in_options = True
            option_text += " " + line.strip()
        elif in_options and line.strip() and not re.match(r'^\d+[\.\)]', line.strip()):
            option_text += " " + line.strip()
    
    if not option_text:
        option_text = normalized_block
    
    # Extract each option
    for label in ['A', 'B', 'C', 'D']:
        pattern = re.compile(
            rf'\(\s*{label}\s*\)\s*(.+?)(?=\s*\(\s*[A-D]\s*\)|$)',
            re.DOTALL
        )
        match = pattern.search(option_text)
        if match:
            opt_text = match.group(1).strip()
            opt_text = re.sub(r'\s+', ' ', opt_text).strip()
            opt_text = re.sub(r'<<<PAGE_\d+>>>', '', opt_text).strip()
            # Remove answer key artifacts like "1-C /2-D/3-A..."
            opt_text = re.sub(r'\d+-[A-D]\s*/\s*\d+-[A-D].*$', '', opt_text).strip()
            if opt_text:
                options[label] = opt_text
    
    return options


def extract_question_text(q_block: str) -> str:
    """Extract the question text (everything before the first option)."""
    # Normalize OCR-misread labels first
    normalized = normalize_option_labels(q_block)
    
    # Find the position of the first option marker in the normalized text
    match = re.search(r'\(\s*[A-D]\s*\)\s*', normalized)
    if match:
        # Use the same position in the original text
        question_text = q_block[:match.start()].strip()
    else:
        question_text = q_block.strip()
    
    return question_text


def clean_text(text: str) -> str:
    """Clean extracted text by removing artifacts and normalizing whitespace."""
    # Remove page markers
    text = re.sub(r'<<<PAGE_\d+>>>', '', text)
    # Remove common OCR artifacts
    text = re.sub(r'[|_]{2,}', '', text)
    # Remove answer key lines
    text = re.sub(r'\d+-[A-D]\s*/\s*\d+-[A-D](?:\s*/\s*\d+-[A-D])*', '', text)
    # Remove watermark text
    text = re.sub(r'WEBSANKUL®?', '', text, flags=re.IGNORECASE)
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    # Remove leading/trailing special characters
    text = text.strip('- •')
    return text


# OUTPUT & VALIDATION

def save_json(questions: list[dict], output_path: str):
    """Save questions to a JSON file with UTF-8 encoding."""
    result = {
        "total_questions": len(questions),
        "questions": questions
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    
    print(f"Saved {len(questions)} questions to: {output_path}")


def save_raw_text(pages_text: list[dict], output_path: str):
    """Save raw OCR text for debugging purposes."""
    with open(output_path, 'w', encoding='utf-8') as f:
        for page in pages_text:
            f.write(f"\n{'='*60}\n")
            f.write(f" PAGE {page['page_number']}\n")
            f.write(f"{'='*60}\n\n")
            f.write(page['text'])
            f.write('\n')
    print(f"   Raw OCR text saved to: {output_path}")


def print_sample(questions: list[dict], count: int = 3):
    """Print sample questions for verification."""
    print(f"\n{'='*60}")
    print(f"SAMPLE OUTPUT (first {min(count, len(questions))} questions)")
    print(f"{'='*60}")
    
    for q in questions[:count]:
        print(f"\nQuestion #{q['question_number']}")
        print(f"Text: {q['question_text'][:120]}")
        if q['options']:
            for key, val in q['options'].items():
                print(f"   ({key}) {val}")
        if 'exam_reference' in q:
            print(f"Exam: {q['exam_reference']}")
        print(f"Page: {q['page_number']}")


def validate_questions(questions: list[dict]) -> dict:
    """Validate parsed questions and report stats."""
    stats = {
        "total": len(questions),
        "with_all_4_options": 0,
        "with_3_options": 0,
        "with_2_options": 0,
        "with_1_option": 0,
        "with_0_options": 0,
        "with_exam_ref": 0,
        "missing_question_text": 0,
    }
    
    for q in questions:
        opt_count = len(q.get("options", {}))
        if opt_count == 4:
            stats["with_all_4_options"] += 1
        elif opt_count == 3:
            stats["with_3_options"] += 1
        elif opt_count == 2:
            stats["with_2_options"] += 1
        elif opt_count == 1:
            stats["with_1_option"] += 1
        else:
            stats["with_0_options"] += 1
        
        if "exam_reference" in q:
            stats["with_exam_ref"] += 1
        
        if not q.get("question_text"):
            stats["missing_question_text"] += 1
    
    print(f"\n{'='*60}")
    print(f"VALIDATION REPORT")
    print(f"{'='*60}")
    print(f"Total questions extracted: {stats['total']}")
    print(f"With all 4 options:     {stats['with_all_4_options']}")
    print(f"With 3 options:         {stats['with_3_options']}")
    print(f"With 2 options:         {stats['with_2_options']}")
    print(f"With 1 option:          {stats['with_1_option']}")
    print(f"With 0 options:          {stats['with_0_options']}")
    print(f"With exam reference:    {stats['with_exam_ref']}")
    print(f"Missing question text:  {stats['missing_question_text']}")
    
    return stats

def main():
    parser = argparse.ArgumentParser(
        description="DocuMorph - Extract Gujarati MCQ questions from scanned PDF to JSON"
    )
    parser.add_argument(
        "pdf_path",
        help="Path to the input PDF file"
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Output JSON file path (default: <pdf_name>_questions.json)"
    )
    parser.add_argument(
        "--pages", "-p",
        default=None,
        help="Page range to process (e.g., '1-10'). Default: all pages"
    )
    parser.add_argument(
        "--engine", "-e",
        choices=["openrouter", "groq", "tesseract", "tesseract+ai"],
        default="openrouter",
        help="OCR engine to use (default: openrouter). 'openrouter' = OpenRouter Vision API, 'groq' = Groq Vision API, 'tesseract' = local Tesseract only, 'tesseract+ai' = Tesseract + AI text correction (cheapest & best hybrid)"
    )
    parser.add_argument(
        "--sample", "-s",
        type=int,
        default=5,
        help="Number of sample questions to display (default: 5)"
    )
    parser.add_argument(
        "--save-raw",
        action="store_true",
        help="Also save raw OCR text for debugging"
    )
    
    args = parser.parse_args()
    
    pdf_path = Path(args.pdf_path)
    
    if not pdf_path.exists():
        print(f"Error: PDF file not found: {pdf_path}")
        sys.exit(1)
    
    # Parse page range
    start_page = None
    end_page = None
    if args.pages:
        parts = args.pages.split('-')
        start_page = int(parts[0])
        end_page = int(parts[1]) if len(parts) > 1 else start_page
    
    # Default output path
    output_path = args.output or f"{pdf_path.stem}_questions.json"
    
    engine_names = {
        "openrouter": f"OpenRouter Vision (cascade: {' → '.join(m.split('/')[-1] for m in OPENROUTER_MODELS)})",
        "groq": "Groq Vision (Llama 4 Scout)",
        "tesseract": "Tesseract OCR",
        "tesseract+ai": "Tesseract + AI Text Correction (Hybrid)"
    }
    engine_name = engine_names.get(args.engine, args.engine)
    
    print(f"\n{'='*60}")
    print(f"DocuMorph - Gujarati PDF Extractor")
    print(f"{'='*60}")
    print(f"Input:  {pdf_path}")
    print(f"Output: {output_path}")
    print(f"Engine: {engine_name}")
    if start_page:
        print(f"Pages:  {start_page} to {end_page}")
    else:
        print(f"Pages:  ALL")
    
    # Step 1: OCR text extraction
    print(f"\nStep 1: OCR Text Extraction ({engine_name})...")
    
    if args.engine == "openrouter":
        pages_text = extract_text_openrouter(str(pdf_path), start_page, end_page)
    elif args.engine == "groq":
        pages_text = extract_text_groq(str(pdf_path), start_page, end_page)
    elif args.engine == "tesseract+ai":
        pages_text = extract_text_tesseract_ai(str(pdf_path), start_page, end_page)
    else:
        pages_text = extract_text_tesseract(str(pdf_path), start_page, end_page)
    
    if not pages_text:
        print("Error: No text could be extracted from the PDF.")
        sys.exit(1)
    
    # Optionally save raw text
    if args.save_raw:
        raw_path = f"{pdf_path.stem}_raw_ocr.txt"
        save_raw_text(pages_text, raw_path)
    
    # Step 2: Parse questions
    print(f"\nStep 2: Parsing questions from OCR text...")
    questions = parse_questions(pages_text)
    
    if not questions:
        print("No questions found. Saving raw OCR text for inspection...")
        raw_path = f"{pdf_path.stem}_raw_ocr.txt"
        save_raw_text(pages_text, raw_path)
        print(f"Check {raw_path} to see what OCR extracted.")
        sys.exit(1)
    
    # Step 3: Validate
    print(f"\nStep 3: Validating...")
    validate_questions(questions)
    
    # Step 4: Show sample
    print_sample(questions, args.sample)
    
    # Step 5: Save JSON
    print(f"\nStep 4: Saving JSON...")
    save_json(questions, output_path)
    
    print(f"\nDone! {len(questions)} questions extracted successfully.")
    print(f"Output: {output_path}")


if __name__ == "__main__":
    main()
