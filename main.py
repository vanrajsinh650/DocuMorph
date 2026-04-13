"""
DocuMorph - Gujarati PDF Question Extractor (OCR-based)
========================================================
Extracts MCQ questions from scanned Gujarati PDF into structured JSON.

Uses Tesseract OCR with Gujarati language support.

Usage:
    python main.py <pdf_path> [--output output.json] [--pages 1-10]

Output JSON format:
    {
        "total_questions": 5500,
        "questions": [
            {
                "id": 1,
                "question_number": "001",
                "question_text": "...",
                "options": {
                    "A": "...",
                    "B": "...",
                    "C": "...",
                    "D": "..."
                },
                "exam_reference": "PI 38/2017-18",
                "page_number": 1
            }
        ]
    }
"""

import sys
import io

# Fix Windows console encoding for Gujarati/Unicode output
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import pytesseract
from pdf2image import convert_from_path
from PIL import Image, ImageFilter, ImageEnhance
import re
import json
import os
import argparse
from pathlib import Path

# ─── CONFIGURATION ──────────────────────────────────────────────────────────
TESSERACT_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
POPPLER_PATH = r"C:\Users\Vanrajsinh\Desktop\DevVault\Building-Hub\DocuMorph\poppler\poppler-24.08.0\Library\bin"

# Set Tesseract path
pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH


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


def ocr_image(image: Image.Image) -> str:
    """Run Tesseract OCR on a preprocessed image and return text."""
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
    OCR each separately, and concatenate left then right.
    This preserves correct question ordering.
    """
    width, height = page_image.size
    
    # Small overlap margin to avoid cutting through text (2% of width)
    margin = int(width * 0.02)
    
    # Crop left column (left half)
    left_col = page_image.crop((0, 0, width // 2 + margin, height))
    
    # Crop right column (right half)
    right_col = page_image.crop((width // 2 - margin, 0, width, height))
    
    # OCR each column separately
    left_text = ocr_image(left_col)
    right_text = ocr_image(right_col)
    
    # Combine: left column first, then right column
    combined_text = left_text.strip() + "\n\n" + right_text.strip()
    
    return {
        "page_number": page_number,
        "text": combined_text
    }


def extract_text_from_pdf(pdf_path: str, start_page: int = None, end_page: int = None) -> list[dict]:
    """
    Convert PDF pages to images and run OCR on each.
    Uses two-column splitting for proper reading order.
    Returns list of {page_number, text} dicts.
    """
    print(f"\n📖 Converting PDF pages to images...")
    
    # Convert PDF pages to images
    kwargs = {
        "pdf_path": pdf_path,
        "dpi": 300,
        "poppler_path": POPPLER_PATH,
        "fmt": "jpeg",
        "thread_count": 4,
    }
    
    if start_page is not None:
        kwargs["first_page"] = start_page
    if end_page is not None:
        kwargs["last_page"] = end_page
    
    images = convert_from_path(**kwargs)
    
    actual_start = start_page or 1
    total = len(images)
    print(f"   Converted {total} pages to images")
    
    # OCR each page (two-column split)
    print(f"\n🔍 Running OCR (Gujarati + English) with 2-column split...")
    pages_text = []
    
    for i, img in enumerate(images):
        page_num = actual_start + i
        result = ocr_page_two_columns(img, page_num)
        pages_text.append(result)
        
        if (i + 1) % 10 == 0 or (i + 1) == total:
            print(f"   OCR completed: {i + 1}/{total} pages")
    
    return pages_text


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
    
    # Pattern to match question numbers like "001)", "002)", "1234)", etc.
    # Also handle OCR variants like "001 )", "O01)" (O instead of 0), etc.
    question_pattern = re.compile(
        r'(?:^|\n)\s*(\d{1,4})\s*\)\s*(.+?)(?=(?:\n\s*\d{1,4}\s*\)\s)|\Z)',
        re.DOTALL
    )
    
    matches = list(question_pattern.finditer(full_text))
    print(f"\n🔍 Found {len(matches)} question blocks")
    
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
    # Pattern 1: Standard format (LETTERS DIGITS/YEAR-YEAR)
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
    # Order matters: do (12) before (1) to avoid partial matches
    # Pattern: (label) at option positions — only replace when it looks like an option marker
    
    # (8) → (B) — very common OCR confusion
    text = re.sub(r'\(8\)\s', '(B) ', text)
    # (3) → (B) — when appearing after (A) context  
    text = re.sub(r'\(3\)\s', '(B) ', text)
    # (૦) → (C) — Gujarati zero confused with C
    text = re.sub(r'\(૦\)\s', '(C) ', text)
    # (0) → (C) — zero confused with C
    text = re.sub(r'\(0\)\s', '(C) ', text)
    # (12) → (D)
    text = re.sub(r'\(12\)\s', '(D) ', text)
    # (2) → (D) — common confusion
    text = re.sub(r'\(2\)\s', '(D) ', text)
    
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
        # Handle OCR spacing variants: (A), ( A ), (A ), ( A)
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
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    # Remove leading/trailing special characters
    text = text.strip('- •')
    return text


def save_json(questions: list[dict], output_path: str):
    """Save questions to a JSON file with UTF-8 encoding."""
    result = {
        "total_questions": len(questions),
        "questions": questions
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    
    print(f"\n✅ Saved {len(questions)} questions to: {output_path}")


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
    print(f"📋 SAMPLE OUTPUT (first {min(count, len(questions))} questions)")
    print(f"{'='*60}")
    
    for q in questions[:count]:
        print(f"\n📌 Question #{q['question_number']}")
        print(f"   Text: {q['question_text'][:120]}")
        if q['options']:
            for key, val in q['options'].items():
                print(f"   ({key}) {val}")
        if 'exam_reference' in q:
            print(f"   📎 Exam: {q['exam_reference']}")
        print(f"   📄 Page: {q['page_number']}")


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
    print(f"📊 VALIDATION REPORT")
    print(f"{'='*60}")
    print(f"   Total questions extracted: {stats['total']}")
    print(f"   ✅ With all 4 options:     {stats['with_all_4_options']}")
    print(f"   ⚠️  With 3 options:         {stats['with_3_options']}")
    print(f"   ⚠️  With 2 options:         {stats['with_2_options']}")
    print(f"   ❌ With 1 option:          {stats['with_1_option']}")
    print(f"   ❌ With 0 options:          {stats['with_0_options']}")
    print(f"   📎 With exam reference:    {stats['with_exam_ref']}")
    print(f"   ❗ Missing question text:  {stats['missing_question_text']}")
    
    return stats


def main():
    parser = argparse.ArgumentParser(
        description="DocuMorph - Extract Gujarati MCQ questions from scanned PDF to JSON (OCR)"
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
        print(f"❌ Error: PDF file not found: {pdf_path}")
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
    
    print(f"\n{'='*60}")
    print(f"🔄 DocuMorph - Gujarati PDF OCR Extractor")
    print(f"{'='*60}")
    print(f"   Input:  {pdf_path}")
    print(f"   Output: {output_path}")
    if start_page:
        print(f"   Pages:  {start_page} to {end_page}")
    else:
        print(f"   Pages:  ALL")
    
    # Step 1: OCR text extraction
    print(f"\n📖 Step 1: OCR Text Extraction...")
    pages_text = extract_text_from_pdf(str(pdf_path), start_page, end_page)
    
    if not pages_text:
        print("❌ Error: No text could be extracted from the PDF.")
        sys.exit(1)
    
    # Optionally save raw text
    if args.save_raw:
        raw_path = f"{pdf_path.stem}_raw_ocr.txt"
        save_raw_text(pages_text, raw_path)
    
    # Step 2: Parse questions
    print(f"\n🔧 Step 2: Parsing questions from OCR text...")
    questions = parse_questions(pages_text)
    
    if not questions:
        print("❌ No questions found. Saving raw OCR text for inspection...")
        raw_path = f"{pdf_path.stem}_raw_ocr.txt"
        save_raw_text(pages_text, raw_path)
        print(f"   Check {raw_path} to see what OCR extracted.")
        sys.exit(1)
    
    # Step 3: Validate
    print(f"\n✔️  Step 3: Validating...")
    validate_questions(questions)
    
    # Step 4: Show sample
    print_sample(questions, args.sample)
    
    # Step 5: Save JSON
    print(f"\n💾 Step 4: Saving JSON...")
    save_json(questions, output_path)
    
    print(f"\n🎉 Done! {len(questions)} questions extracted successfully.")
    print(f"   Output: {output_path}")


if __name__ == "__main__":
    main()
