"""
DocuMorph - PDF Text Inspector
===============================
Use this to debug/inspect raw text extracted from your PDF.
Helps you verify the text extraction quality before full parsing.

Usage:
    python inspect_pdf.py <pdf_path> [--pages 1-5]
"""

import pdfplumber
import sys
import argparse
from pathlib import Path


def inspect_pdf(pdf_path: str, page_range: str = None):
    """Inspect raw text from PDF pages."""
    
    with pdfplumber.open(pdf_path) as pdf:
        total_pages = len(pdf.pages)
        print(f"\nPDF Info:")
        print(f"File: {pdf_path}")
        print(f"Total pages: {total_pages}")
        
        # Determine which pages to inspect
        if page_range:
            parts = page_range.split('-')
            start_page = int(parts[0]) - 1
            end_page = int(parts[1]) if len(parts) > 1 else start_page + 1
        else:
            start_page = 0
            end_page = min(3, total_pages)  # Default: first 3 pages
        
        print(f"Inspecting pages: {start_page + 1} to {end_page}")
        
        for i in range(start_page, end_page):
            page = pdf.pages[i]
            text = page.extract_text()
            
            print(f"\n{'='*70}")
            print(f"PAGE {i + 1}")
            print(f"{'='*70}")
            
            if text:
                print(text)
            else:
                print("No text extracted from this page!")
                print("This might be a scanned/image-based page.")
                
                # Check if page has images
                if page.images:
                    print(f"Found {len(page.images)} images on this page.")
                    print("You may need OCR (Tesseract) for this PDF.")
        
        # Quick stats
        print(f"\n{'='*70}")
        print(f"Quick Stats:")
        
        text_pages = 0
        empty_pages = 0
        total_chars = 0
        
        for page in pdf.pages:
            text = page.extract_text()
            if text and len(text.strip()) > 10:
                text_pages += 1
                total_chars += len(text)
            else:
                empty_pages += 1
        
        print(f"Pages with text: {text_pages}")
        print(f"Empty/image pages: {empty_pages}")
        print(f"Total characters: {total_chars:,}")
        
        if empty_pages > text_pages:
            print(f"\nWARNING: Most pages have no extractable text!")
            print(f"This PDF is likely scanned/image-based.")
            print(f"You'll need OCR extraction instead.")
        elif text_pages > 0:
            print(f"\nPDF has extractable text. Ready for parsing!")


def main():
    parser = argparse.ArgumentParser(
        description="DocuMorph - Inspect PDF text extraction quality"
    )
    parser.add_argument("pdf_path", help="Path to the PDF file")
    parser.add_argument(
        "--pages", "-p",
        default=None,
        help="Page range to inspect (e.g., '1-5'). Default: first 3 pages"
    )
    
    args = parser.parse_args()
    
    pdf_path = Path(args.pdf_path)
    if not pdf_path.exists():
        print(f"Error: File not found: {pdf_path}")
        sys.exit(1)
    
    inspect_pdf(str(pdf_path), args.pages)


if __name__ == "__main__":
    main()
