# DocuMorph

A high-accuracy Document Image Analysis (DIA) and Optical Character Recognition (OCR) pipeline engineered to extract complex, multi-column Gujarati text from scanned PDF artifacts and serialize the output into structured JSON.

## Architecture

DocuMorph is designed to handle the structural complexities of scanned examination papers, specifically addressing the documented failure rates of traditional OCR engines when processing Indic scripts (e.g., character mangling, ligature misinterpretation).

### Vision-Based OCR Engine
The primary extraction pipeline interfaces with the Groq inference API, utilizing multimodal large language models (`meta-llama/llama-4-scout-17b-16e-instruct`). This circumvents conventional bounding-box and character-recognition strategies, instead allowing the model to interpret document layout and script natively as a unified visual-spatial task.

### Multi-Key Load Balancing
To mitigate rate-limiting constraints on the Groq free tier API (HTTP 429), the system implements a `GroqKeyPool` coordinator with reliability-first behavior. It rotates keys in round-robin order, applies global request pacing, tracks per-key cooldown state, and uses adaptive exponential backoff (+ jitter) when 429 pressure is high. If all keys are cooling down, it waits for earliest recovery instead of hot-looping.

### Memory Optimization
Image rasterization via Poppler (`pdf2image`) operates strictly sequentially. Converting high-resolution PDFs (300 DPI) to intermediate bitmap arrays can cause immediate Out-Of-Memory (OOM) faults within constrained execution environments (e.g., 1GB container limits). The system mitigates this by isolating the conversion, inference, and memory deallocation lifecycle to a single-page scope. 

### Tesseract Fallback and Heuristic Segmentation
An offline OCR fallback is preserved via `pytesseract`. To resolve dual-column text flow lacking native digital structure, the pipeline applies deterministic geometric segmentation. The system calculates continuous mid-point vertical vectors across the raster image with safety margins, isolating left and right sub-regions before applying localized character recognition.

## System Dependencies

The application relies on binary dependencies for PDF rendering and offline OCR.

### Linux (Streamlit Cloud, Ubuntu/Debian)
Ensure the following system packages are installed:
```bash
sudo apt-get update
sudo apt-get install poppler-utils tesseract-ocr tesseract-ocr-guj
```

### Windows
1. Install [Tesseract OCR](https://github.com/UB-Mannheim/tesseract/wiki). Ensure the Gujarati language data file (`guj.traineddata`) is present in the `tessdata` directory.
2. Download [Poppler for Windows](https://github.com/oschwartz10612/poppler-windows/releases/), extract, and configure the path in the local environment.

## Installation

```bash
git clone https://github.com/vanrajsinh650/DocuMorph.git
cd DocuMorph
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Configuration

API keys must be securely provisioned for the primary Groq engine.

### Local Execution (.env)
Create a `.env` file in the root directory:
```
GROQ_API_KEY_1="gsk_XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"
GROQ_API_KEY_2="gsk_YYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYY"
```

### Streamlit Cloud (st.secrets)
Configure the TOML formatted secrets via the application deployment dashboard:
```toml
GROQ_API_KEY_1 = "gsk_XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"
GROQ_API_KEY_2 = "gsk_YYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYY"
```

## Usage

### Web Interface (Streamlit)
To initialize the user interface and local server:
```bash
streamlit run app.py
```
The interface now runs an automatic two-stage pipeline:
1. Tesseract OCR (raw extraction)
2. Groq text correction (Gujarati repair)

It then provides both downloads:
- corrected JSON (`*_questions_fixed.json`)
- raw JSON (`*_questions_raw.json`)

### Command Line Interface
Execute the core binary directly for headless operations:
```bash
python main.py path/to/document.pdf --pages 10-20 --engine tesseract+groq
```
Arguments:
* `--pages`: Specify a continuous sub-range (e.g., 1-5). Defaults to the entire document.
* `--engine`: Select execution layer (`openrouter`, `groq`, `tesseract`, `tesseract+ai`, `tesseract+groq`). Default is `openrouter`.
* `--output`: Declare the target output path. In `tesseract+groq`, this is the fixed JSON path.
* `--save-raw`: Retain raw intermediate text prior to RegEx parsing.

For `--engine tesseract+groq`, two JSON files are produced:
- raw JSON: `<pdf_stem>_questions_raw.json`
- fixed JSON: `<pdf_stem>_questions_fixed.json` (or `--output` path)

## Output Schema

The parser extracts hierarchical data structures via compiled regular expressions, enforcing strict validation on expected metadata keys.

```json
{
  "total_questions": 1,
  "questions": [
    {
      "id": 1,
      "question_number": "001",
      "question_text": "ગુજરાતી ભાષાનો પ્રશ્ન...",
      "options": {
        "A": "વિકલ્પ એક",
        "B": "વિકલ્પ બે",
        "C": "વિકલ્પ ત્રણ",
        "D": "વિકલ્પ ચાર"
      },
      "exam_reference": "PI 38/2017-18",
      "page_number": 1
    }
  ]
}
```
