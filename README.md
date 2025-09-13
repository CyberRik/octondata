# Unstructured Data Parsing Pipeline

A comprehensive Python-based pipeline for extracting and processing unstructured data from PDF documents, with integrated RAG (Retrieve and Generate) capabilities for intelligent document retrieval and analysis.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Image Extraction Logic](#image-extraction-logic)
- [RAG Pipeline Integration](#rag-pipeline-integration)
- [Testing](#testing)
- [Output Structure](#output-structure)
- [API Reference](#api-reference)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)

## Overview

The Unstructured Data Parsing Pipeline is designed to extract, process, and organize content from PDF documents into structured formats. The pipeline focuses on three primary content types: embedded images, tables, and text. It incorporates advanced computer vision techniques for intelligent image filtering and integrates with a RAG system for document retrieval and generation tasks.

### Key Capabilities

- **PDF Document Processing**: Comprehensive extraction of images, tables, and text from PDF files
- **Embedded Image Detection**: Advanced filtering to extract only truly embedded images, excluding page elements and table crops
- **Table Extraction**: Conversion of table data into structured formats (CSV and JSON)
- **Text Processing**: Extraction and cleaning of textual content with OCR support
- **RAG Integration**: Document indexing and intelligent retrieval for question-answering tasks
- **Dynamic Content Handling**: Automatic adaptation to different document structures and naming conventions

## Features

### PDF Parsing
- **Multi-format Support**: Handles various PDF types including scanned documents
- **OCR Integration**: Extracts text from image-based PDFs using Tesseract
- **Handwritten Content Detection**: Identifies and processes handwritten text
- **Document Structure Analysis**: Understands document layout and organization

### Embedded Image Extraction
- **Computer Vision Analysis**: Uses OpenCV for advanced image content analysis
- **Size-based Filtering**: Excludes full-page images and small UI elements
- **Aspect Ratio Validation**: Prefers images with reasonable proportions
- **Color Content Analysis**: Identifies images with rich visual content
- **Edge Detection**: Analyzes image complexity and structure
- **Dynamic Selection**: Automatically selects the best embedded image when multiple candidates exist

### Table Processing
- **Multi-engine Extraction**: Uses multiple table extraction methods (img2table, tabula, camelot)
- **Data Cleaning**: Removes duplicates and validates table content
- **Format Conversion**: Exports tables in both CSV and JSON formats
- **Structure Validation**: Ensures proper table formatting and completeness

### Text Extraction
- **Multi-source Text**: Combines text from various sources (OCR, native text, handwritten)
- **Content Cleaning**: Removes artifacts and normalizes text formatting
- **Chunking Support**: Splits large text into manageable chunks for processing
- **Encoding Handling**: Robust handling of various text encodings

### RAG Pipeline Integration
- **Document Indexing**: Creates searchable indexes from extracted content
- **TF-IDF Vectorization**: Implements semantic document retrieval
- **Query Processing**: Handles natural language queries about document content
- **Multi-modal Support**: Processes text, table, and image metadata
- **Confidence Scoring**: Provides relevance scores for retrieved content

## Installation

### Prerequisites

- Python 3.8 or higher
- Tesseract OCR (for text extraction from images)
- Git (for installing od-parse)

### System Dependencies

#### Windows
```bash
# Download and install Tesseract from:
# https://github.com/UB-Mannheim/tesseract/wiki
# Add Tesseract to your system PATH
```

#### macOS
```bash
brew install tesseract
```

#### Linux (Ubuntu/Debian)
```bash
sudo apt-get update
sudo apt-get install tesseract-ocr
```

### Python Dependencies

1. **Clone the repository**:
```bash
git clone <repository-url>
cd octondata1
```

2. **Install Python dependencies**:
```bash
pip install -r requirements.txt
```

3. **Verify installation**:
```bash
python test_enhanced_pipeline.py
```

### Dependencies Overview

The pipeline requires several key libraries:

- **od-parse**: Core PDF parsing functionality
- **OpenCV**: Computer vision for image analysis
- **Pandas**: Data manipulation for table processing
- **scikit-learn**: Machine learning for RAG functionality
- **PyYAML**: Configuration file handling
- **Pillow**: Image processing
- **Tesseract**: OCR for text extraction

## Configuration

The pipeline uses a YAML configuration file (`streamlined_config.yaml`) for customization:

### Basic Configuration

```yaml
# Input and Output Settings
input_file: "./sample.pdf"
output_dir: "./outputs"

# File Naming Settings
use_timestamps: false
preserve_source_name: true
```

### Parser Configuration

```yaml
parser:
  pdf: true
  pdf_config:
    extract_tables: true
    extract_forms: true
    extract_structure: true
    extract_images: true
    ocr: true
    handwritten_detection: true
    img2table: true
    image_naming: "descriptive"
```

### RAG Configuration

```yaml
rag:
  index_path: "./rag_index.json"
  chunk_size: 500
  chunk_overlap: 50
  top_k: 3
  enable_rag: true
```

### Logging Configuration

```yaml
logging:
  enable_logging: true
  log_file: "pipeline.log"
  log_level: "INFO"
```

## Usage

### Command Line Interface

#### Basic PDF Processing
```bash
python enhanced_refined_pipeline.py --input document.pdf --output ./results
```

#### With Custom Configuration
```bash
python enhanced_refined_pipeline.py --input document.pdf --config custom_config.yaml
```

#### With Timestamps
```bash
python enhanced_refined_pipeline.py --input document.pdf --timestamps
```

#### RAG Pipeline Usage
```bash
# Build RAG index
python rag_pipeline.py build --pdf document.pdf

# Query the document
python rag_pipeline.py query --question "What images are in the document?"

# Get index statistics
python rag_pipeline.py stats
```

### Python API

#### Basic Usage
```python
from enhanced_refined_pipeline import EnhancedRefinedPipeline

# Initialize pipeline
pipeline = EnhancedRefinedPipeline()

# Process a PDF
result = pipeline.process_document("document.pdf")

if result['status'] == 'success':
    print(f"Processing completed in {result['processing_time_seconds']:.2f}s")
    print(f"Extracted: {result['content_extracted']}")
    
    # Access specific content
    embedded_images = result['images']
    tables = result['tables']
    text = result['text']
```

#### RAG Pipeline Usage
```python
from rag_pipeline import RAGPipeline

# Initialize RAG pipeline
rag = RAGPipeline()

# Build index from PDF
result = rag.process_document_and_build_index("document.pdf")
if result['status'] == 'success':
    print(f"Index built with {result['documents_count']} documents")

# Query the document
query_result = rag.query("What is in the table?")
if query_result['status'] == 'success':
    print(f"Answer: {query_result['answer']}")
    print(f"Confidence: {query_result['confidence']:.3f}")
```

## Image Extraction Logic

The pipeline implements sophisticated image filtering to extract only truly embedded images:

### Analysis Process

1. **Size Analysis**: Filters images based on dimensions
   - Excludes very large images (likely full pages)
   - Excludes very small images (UI elements)
   - Prefers medium-sized images (300-1500px width, 300-1200px height)

2. **Aspect Ratio Validation**: Ensures reasonable proportions
   - Prefers aspect ratios between 0.5 and 2.0
   - Excludes very wide or very tall images (likely tables or banners)

3. **Color Content Analysis**: Evaluates visual richness
   - Calculates color variance to identify rich content
   - Prefers images with high color variance
   - Excludes low-variance images (likely text or simple graphics)

4. **Edge Detection**: Analyzes image complexity
   - Uses Canny edge detection to identify clear edges
   - Prefers images with good edge density
   - Excludes images with very few edges

5. **File Size Validation**: Ensures reasonable file sizes
   - Prefers files between 0.1MB and 2.0MB
   - Excludes very large files (likely full pages)
   - Excludes very small files (likely simple graphics)

6. **Best Selection Algorithm**: When multiple candidates exist
   - Calculates composite scores based on all factors
   - Selects the highest-scoring image
   - Provides detailed confidence metrics

### Dynamic Handling

The system dynamically handles different PDF structures:
- **Filename Independence**: Works regardless of image naming conventions
- **Content-based Selection**: Uses image characteristics rather than filenames
- **Adaptive Filtering**: Adjusts criteria based on document content
- **Confidence Scoring**: Provides detailed analysis of why images were selected or filtered

## RAG Pipeline Integration

The RAG (Retrieve and Generate) pipeline provides intelligent document retrieval:

### Index Building Process

1. **Content Extraction**: Processes PDF using the main pipeline
2. **Text Chunking**: Splits text into overlapping chunks for better retrieval
3. **Multi-modal Processing**: Handles text, table, and image metadata
4. **TF-IDF Vectorization**: Creates searchable document vectors
5. **Index Storage**: Saves index for future queries

### Query Processing

1. **Query Vectorization**: Converts natural language queries to vectors
2. **Similarity Search**: Finds most relevant document chunks
3. **Response Generation**: Combines retrieved content into coherent answers
4. **Confidence Scoring**: Provides relevance scores for retrieved content

### Supported Query Types

- **Content Questions**: "What is in the table?"
- **Image Queries**: "What images are in the document?"
- **General Questions**: "What is the main content?"
- **Specific Searches**: "Find information about [topic]"

## Testing

### Running Tests

```bash
# Run all tests
python test_enhanced_pipeline.py

# Run specific test categories
python -c "from test_enhanced_pipeline import test_enhanced_image_parsing; test_enhanced_image_parsing()"
```

### Test Coverage

The test suite covers:

- **Image Parsing**: Validates embedded image extraction and filtering
- **Table Extraction**: Ensures correct table data extraction and formatting
- **Text Processing**: Verifies text extraction and cleaning
- **RAG Integration**: Tests document indexing and querying
- **Output Organization**: Validates file structure and naming
- **Error Handling**: Tests edge cases and error scenarios

### Test Results

```
Test Summary
============================================================
Enhanced Image Parsing     PASS
RAG Integration            PASS
Embedded Image Detection   PASS
Output Organization        PASS
Error Handling             PASS

Overall: 5/5 tests passed
```

## Output Structure

The pipeline creates organized output directories:

```
enhanced_output/
├── images/                    # Extracted embedded images
│   └── sample_embedded_4.png  # Best embedded image selected
├── tables/                    # Extracted tables
│   ├── sample_table.csv       # Table in CSV format
│   └── sample_table.json      # Table with metadata
├── text/                      # Extracted text
│   └── sample_extracted_text.txt
└── processing_summary_sample.json  # Processing metadata
```

### File Naming Convention

- **Images**: `{document_name}_embedded_{number}.{extension}`
- **Tables**: `{document_name}_table.{extension}`
- **Text**: `{document_name}_extracted_text.txt`
- **Metadata**: `processing_summary_{document_name}.json`

## API Reference

### EnhancedRefinedPipeline Class

#### Methods

- `__init__(config_path: str)`: Initialize pipeline with configuration
- `process_document(pdf_path: str)`: Process a PDF document
- `extract_embedded_images(parsed_data: Dict, source_file: str)`: Extract embedded images
- `extract_tables(parsed_data: Dict, source_file: str)`: Extract and process tables
- `extract_text(parsed_data: Dict, source_file: str)`: Extract and clean text
- `get_processing_stats()`: Get processing statistics

### RAGPipeline Class

#### Methods

- `__init__(config_path: str)`: Initialize RAG pipeline
- `process_document_and_build_index(pdf_path: str)`: Build searchable index
- `query(question: str, top_k: int)`: Query the document
- `load_index(index_path: str)`: Load existing index
- `get_index_stats()`: Get index statistics

## Troubleshooting

### Common Issues

#### Tesseract Not Found
```
Error: tesseract is not installed or it's not in your PATH
```
**Solution**: Install Tesseract and ensure it's in your system PATH

#### Memory Issues with Large PDFs
```
Error: Out of memory during processing
```
**Solution**: Process smaller batches or increase system memory

#### Permission Errors
```
Error: Permission denied when writing to output directory
```
**Solution**: Ensure write permissions for the output directory

#### Import Errors
```
Error: No module named 'od_parse'
```
**Solution**: Install dependencies with `pip install -r requirements.txt`

### Debug Mode

Enable detailed logging for troubleshooting:

```python
import logging
logging.getLogger().setLevel(logging.DEBUG)

pipeline = EnhancedRefinedPipeline()
result = pipeline.process_document("document.pdf")
```

### Performance Optimization

- **Large PDFs**: Process in smaller batches
- **Memory Usage**: Monitor RAM usage during processing
- **Storage**: Ensure sufficient disk space for output files
- **CPU**: Multi-core processing is supported for parallel operations

