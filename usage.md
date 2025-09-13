# Quick Usage Guide

## Prerequisites

1. **Install Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Install Tesseract OCR** (required for text extraction):
   - **Windows**: Download from [GitHub](https://github.com/UB-Mannheim/tesseract/wiki)
   - **macOS**: `brew install tesseract`
   - **Linux**: `sudo apt-get install tesseract-ocr`

## Basic Usage

### 1. Process a PDF Document
```bash
python enhanced_refined_pipeline.py --input your_document.pdf --output ./results
```

### 2. Use RAG Pipeline
```bash
# Build searchable index
python rag_pipeline.py build --pdf your_document.pdf

# Query the document
python rag_pipeline.py query --question "What is in the table?"
```

### 3. Python API
```python
from enhanced_refined_pipeline import EnhancedRefinedPipeline

# Process PDF
pipeline = EnhancedRefinedPipeline()
result = pipeline.process_document("document.pdf")

# Access results
images = result['images']
tables = result['tables'] 
text = result['text']
```

## Output Structure
```
results/
├── images/          # Extracted embedded images
├── tables/          # Tables in CSV/JSON format
└── text/            # Extracted text content
```

## Configuration
Edit `streamlined_config.yaml` to customize:
- Input/output paths
- Extraction settings
- RAG parameters
- Logging options

## Troubleshooting
- **Tesseract not found**: Ensure Tesseract is installed and in PATH
- **Permission errors**: Check write permissions for output directory
- **Import errors**: Run `pip install -r requirements.txt`

For detailed documentation, see [README.md](README.md).
