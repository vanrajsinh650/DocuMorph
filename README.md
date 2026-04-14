# DocuMorph

A high-accuracy Document Image Analysis (DIA) and Optical Character Recognition (OCR) pipeline engineered to extract complex, multi-column Gujarati text from scanned PDF artifacts and serialize the output into structured JSON.

## Architecture

DocuMorph is designed to handle the structural complexities of scanned examination papers, specifically addressing the documented failure rates of traditional OCR engines when processing Indic scripts (e.g., character mangling, ligature misinterpretation).

### Vision-Based OCR Engine
The primary extraction pipeline interfaces with the Groq inference API, utilizing multimodal large language models (`meta-llama/llama-4-scout-17b-16e-instruct`). This circumvents conventional bounding-box and character-recognition strategies, instead allowing the model to interpret document layout and script natively as a unified visual-spatial task.

### Multi-Key Load Balancing
To mitigate rate-limiting constraints on the Groq free tier API (HTTP 429), the system implements a `GroqKeyPool` coordinator. This class maintains a connection pool of multiple API keys, executing requests via a round-robin algorithm. Upon encountering a rate limit, the pool registers a localized cooldown state and automatically falls back to the next available unexhausted key, ensuring continuous pipeline execution.

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
The interface manages file I/O, configures extraction bounds, and maintains asynchronous state, preserving JSON output binaries in `st.session_state` to prevent DOM-reconciliation data loss.

### Command Line Interface
Execute the core binary directly for headless operations:
```bash
python main.py path/to/document.pdf --pages 10-20 --engine groq --output results.json
```
Arguments:
* `--pages`: Specify a continuous sub-range (e.g., 1-5). Defaults to the entire document.
* `--engine`: Select the execution layer (`groq` or `tesseract`). Defaults to `groq`.
* `--output`: Declare the target relative path for serialized JSON.
* `--save-raw`: Retain raw intermediate text prior to RegEx parsing.

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
