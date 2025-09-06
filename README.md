## Advanced RAG Pipeline using od-parse

This project demonstrates a comprehensive unstructured data pipeline using the open-source `od-parse` library for advanced parsing, and a pure-Python TF-IDF retriever for RAG-style retrieval.

Source for `od-parse`: [octondata/od-parse](https://github.com/octondata/od-parse)

### Features
- **Advanced PDF Parsing**: Extract text, tables, forms, document structure, and handwritten content
- **Comprehensive OCR**: Built-in OCR with handwritten text detection
- **Smart Image Naming**: Descriptive naming for extracted images (text_, table_, image_ prefixes)
- **RAG Pipeline**: Chunk text into overlapping segments and build TF-IDF index
- **Multiple Output Formats**: Text summary and structured JSON output
- **Dependency-Light**: Pure Python implementation with minimal external dependencies

### Requirements
- Python 3.13 (tested)
- Tesseract installed and available on PATH if you need OCR for images

Install deps:

```bash
pip install -r requirements.txt
```

### Configure
Edit `config.yaml`:
- `input_file`: document path for `run_pipeline.py`
- `output_dir`: output directory for generation/logs
- `rag`: settings for the RAG index (data directory, index path, chunking, top_k)

Example defaults:

```yaml
input_file: "./sample.pdf"
output_dir: "./outputs"
rag:
  data_dir: "./data"
  index_path: "./rag_index.json"
  chunk_size: 500
  chunk_overlap: 50
  top_k: 3
```

### Usage

1) Build index from a directory of documents:

```bash
python rag_pipeline.py build --config config.yaml --data-dir ./data --index-path ./rag_index.json
```

2) Query the index:

```bash
python rag_pipeline.py query --config config.yaml --index-path ./rag_index.json --question "What is the invoice total?"
```

3) Parse a single document with advanced features:

```bash
python run_pipeline.py
```

This will parse the document at `config.yaml.input_file` using advanced `od-parse` features and create:

**Outputs:**
- `outputs/parsed_text.txt` - Comprehensive text with all extracted content:
  - Main document text
  - OCR extracted content (handwritten text, images)
  - Extracted tables in markdown format
  - Form fields and their values
  - Document structure (headings, paragraphs, lists)
- `outputs/parsed_data.json` - Structured JSON with all extraction results

**Advanced Features:**
- **Tables**: Neural network-based table extraction with markdown output
- **Forms**: Automatic form field detection and value extraction
- **Structure**: Document hierarchy with headings, paragraphs, lists
- **OCR**: Handwritten text detection and image OCR
- **Smart Naming**: Extracted images use descriptive prefixes (text_, table_, image_)

### Notes
- The TF-IDF implementation is intentionally simple and dependency-light to maximize compatibility on Windows + Python 3.13.
- For production-grade generation (LLM) or embedding retrieval, you can integrate your preferred model service in `run_pipeline.py` and/or replace TF-IDF with FAISS or a hosted vector DB.

### License
MIT



