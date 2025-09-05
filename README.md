## Minimal RAG Pipeline using od-parse

This project demonstrates a lightweight, dependency-friendly unstructured data pipeline using the open-source `od-parse` library for parsing, and a pure-Python TF-IDF retriever for RAG-style retrieval.

Source for `od-parse`: [octondata/od-parse](https://github.com/octondata/od-parse)

### Features
- Parse PDFs via `od-parse` and images via Tesseract OCR
- Chunk text into overlapping segments
- Build a TF-IDF index (no Rust/CUDA requirements)
- Query the index from the CLI and view top-K relevant chunks

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

3) Parse a single document:

```bash
python run_pipeline.py
```

This will parse the document at `config.yaml.input_file` and write the full extracted text to `outputs/parsed_text.txt`. If tables are detected, they are saved to `outputs/extracted_tables.md`.

### Notes
- The TF-IDF implementation is intentionally simple and dependency-light to maximize compatibility on Windows + Python 3.13.
- For production-grade generation (LLM) or embedding retrieval, you can integrate your preferred model service in `run_pipeline.py` and/or replace TF-IDF with FAISS or a hosted vector DB.

### License
MIT



