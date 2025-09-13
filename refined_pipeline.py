#!/usr/bin/env python3
"""
Refined PDF Parsing Pipeline
Fixed version that correctly extracts tables and filters relevant images
"""

import os
import json
import logging
import yaml
import hashlib
import re
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
import cv2
import numpy as np
from PIL import Image

# od-parse integration
from od_parse import parse_pdf


class RefinedPipeline:
    """Refined PDF parsing pipeline with improved table and image extraction"""
    
    def __init__(self, config_path: str = "streamlined_config.yaml"):
        self.config = self._load_config(config_path)
        self._setup_logging()
        
        # Output directory structure
        self.base_output_dir = Path(self.config.get('output_dir', './outputs'))
        self.images_dir = self.base_output_dir / 'images'
        self.tables_dir = self.base_output_dir / 'tables'
        self.text_dir = self.base_output_dir / 'text'
        
        # Create output directories
        self._create_output_directories()
        
        # File naming configuration
        self.use_timestamps = self.config.get('use_timestamps', False)
        self.preserve_source_name = self.config.get('preserve_source_name', True)
        
        # Content filtering
        self.min_image_size = (100, 100)  # Minimum image dimensions
        self.max_image_size = (5000, 5000)  # Maximum image dimensions
        self.table_confidence_threshold = 0.5
        
        logging.info("Refined pipeline initialized successfully")
    
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file"""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logging.warning(f"Failed to load config from {config_path}: {e}")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict:
        """Get default configuration"""
        return {
            'output_dir': './outputs',
            'use_timestamps': False,
            'preserve_source_name': True,
            'parser': {
                'pdf': True,
                'pdf_config': {
                    'extract_tables': True,
                    'extract_forms': True,
                    'extract_structure': True,
                    'extract_images': True,
                    'ocr': True,
                    'handwritten_detection': True,
                    'img2table': True,
                    'image_naming': 'descriptive'
                }
            },
            'logging': {
                'enable_logging': True,
                'log_file': 'refined_pipeline.log',
                'log_level': 'INFO'
            }
        }
    
    def _setup_logging(self):
        """Setup logging configuration"""
        log_config = self.config.get('logging', {})
        if log_config.get('enable_logging', True):
            log_file = log_config.get('log_file', 'refined_pipeline.log')
            log_level = getattr(logging, log_config.get('log_level', 'INFO'))
            logging.basicConfig(
                filename=log_file,
                level=log_level,
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
        else:
            logging.basicConfig(level=logging.INFO)
    
    def _create_output_directories(self):
        """Create output directory structure"""
        for directory in [self.base_output_dir, self.images_dir, self.tables_dir, self.text_dir]:
            directory.mkdir(parents=True, exist_ok=True)
        logging.info(f"Output directories created at: {self.base_output_dir}")
    
    def _generate_unique_filename(self, base_name: str, extension: str, 
                                source_file: str = None, content_type: str = None) -> str:
        """Generate unique filename to prevent overwriting"""
        # Extract source file name if provided
        if source_file and self.preserve_source_name:
            source_name = Path(source_file).stem
            base_name = f"{source_name}_{base_name}"
        
        # Add timestamp if enabled
        if self.use_timestamps:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_name = f"{base_name}_{timestamp}"
        
        # Ensure unique filename
        counter = 1
        original_name = base_name
        while True:
            filename = f"{base_name}.{extension}"
            target_path = getattr(self, f"{content_type}_dir") / filename
            if not target_path.exists():
                return filename
            base_name = f"{original_name}_{counter}"
            counter += 1
    
    def _calculate_content_hash(self, content: Any) -> str:
        """Calculate hash for content deduplication"""
        if isinstance(content, (str, bytes)):
            content_str = content if isinstance(content, str) else content.decode('utf-8')
        else:
            content_str = json.dumps(content, sort_keys=True, default=str)
        
        return hashlib.md5(content_str.encode('utf-8')).hexdigest()
    
    def _is_relevant_image(self, image_path: str) -> bool:
        """Determine if an image is relevant content (not a table crop or small element)"""
        try:
            with Image.open(image_path) as img:
                width, height = img.size
                
                # Check image dimensions
                if width < self.min_image_size[0] or height < self.min_image_size[1]:
                    logging.debug(f"Image too small: {image_path} ({width}x{height})")
                    return False
                
                if width > self.max_image_size[0] or height > self.max_image_size[1]:
                    logging.debug(f"Image too large: {image_path} ({width}x{height})")
                    return False
                
                # Check if it's likely a table crop (very wide or very tall)
                aspect_ratio = width / height
                if aspect_ratio > 5 or aspect_ratio < 0.2:
                    logging.debug(f"Image aspect ratio suggests table crop: {image_path} ({aspect_ratio:.2f})")
                    return False
                
                # Check filename patterns that suggest table crops
                filename = Path(image_path).name.lower()
                if any(pattern in filename for pattern in ['crop', 'table', 'img_']):
                    logging.debug(f"Image filename suggests table crop: {image_path}")
                    return False
                
                # Check if it's a full page image (likely relevant content)
                if 'page_' in filename and 'img_' not in filename:
                    logging.debug(f"Full page image detected: {image_path}")
                    return True
                
                # Additional checks for content relevance
                # Load image for analysis
                img_array = np.array(img)
                
                # Check for sufficient color variation (not just text)
                if len(img_array.shape) == 3:
                    color_variance = np.var(img_array)
                    if color_variance < 100:  # Very low variance suggests mostly text
                        logging.debug(f"Image has low color variance: {image_path}")
                        return False
                
                logging.debug(f"Image passed relevance checks: {image_path}")
                return True
                
        except Exception as e:
            logging.warning(f"Failed to analyze image {image_path}: {e}")
            return False
    
    def _clean_table_data(self, table_data: List[Dict]) -> List[Dict]:
        """Clean and validate table data"""
        if not table_data:
            return []
        
        cleaned_data = []
        seen_rows = set()
        
        for row in table_data:
            if not isinstance(row, dict):
                continue
            
            # Extract values and clean them
            values = []
            for key, value in row.items():
                if isinstance(value, str):
                    # Clean string values
                    cleaned_value = value.strip()
                    # Remove extra whitespace and newlines
                    cleaned_value = re.sub(r'\s+', ' ', cleaned_value)
                    # Remove empty or meaningless values
                    if cleaned_value and cleaned_value not in ['', ' ', 'nan', 'NaN', 'None']:
                        values.append(cleaned_value)
                elif value is not None:
                    values.append(str(value).strip())
            
            # Only include rows with meaningful content
            if values and len(values) > 1:  # At least 2 columns
                # Create a hash of the row content for deduplication
                row_hash = self._calculate_content_hash(values)
                if row_hash not in seen_rows:
                    seen_rows.add(row_hash)
                    cleaned_data.append(values)
        
        return cleaned_data
    
    def _extract_clean_table(self, parsed_data: Dict[str, Any]) -> Optional[List[List[str]]]:
        """Extract and clean the main table from parsed data"""
        try:
            # Get tables from various sources
            table_sources = [
                parsed_data.get('tables', []),
                parsed_data.get('extracted_tables', []),
                parsed_data.get('table_data', [])
            ]
            
            all_tables = []
            for source in table_sources:
                if isinstance(source, list):
                    all_tables.extend(source)
                elif isinstance(source, dict):
                    all_tables.append(source)
            
            logging.info(f"Found {len(all_tables)} table sources to analyze")
            
            # Find the best table (most complete, highest confidence)
            best_table = None
            best_score = 0
            
            for table_data in all_tables:
                try:
                    if isinstance(table_data, dict):
                        table_rows = table_data.get('data', table_data.get('rows', []))
                        confidence = table_data.get('confidence', 0.0)
                    elif isinstance(table_data, list):
                        table_rows = table_data
                        confidence = 1.0
                    else:
                        continue
                    
                    if not table_rows:
                        continue
                    
                    # Clean the table data
                    cleaned_rows = self._clean_table_data(table_rows)
                    
                    if not cleaned_rows:
                        continue
                    
                    # Calculate table quality score
                    row_count = len(cleaned_rows)
                    col_count = max(len(row) for row in cleaned_rows) if cleaned_rows else 0
                    
                    # Score based on size, confidence, and content quality
                    score = (row_count * col_count * confidence) / 100
                    
                    # Prefer tables with reasonable dimensions
                    if 2 <= row_count <= 20 and 2 <= col_count <= 10:
                        score *= 1.5
                    
                    logging.debug(f"Table candidate: {row_count} rows, {col_count} cols, confidence: {confidence:.2f}, score: {score:.2f}")
                    
                    if score > best_score:
                        best_score = score
                        best_table = cleaned_rows
                        
                except Exception as e:
                    logging.warning(f"Failed to process table candidate: {e}")
                    continue
            
            if best_table:
                logging.info(f"Selected best table with {len(best_table)} rows and score {best_score:.2f}")
                return best_table
            else:
                logging.warning("No suitable table found")
                return None
                
        except Exception as e:
            logging.error(f"Failed to extract clean table: {e}")
            return None
    
    def parse_pdf(self, pdf_path: str) -> Dict[str, Any]:
        """Parse PDF using od-parse with comprehensive extraction"""
        try:
            logging.info(f"Starting PDF parsing: {pdf_path}")
            
            # Parse with od-parse using full pipeline
            parsed_data = parse_pdf(
                file_path=pdf_path,
                output_format="json",
                pipeline_type="full",
                use_deep_learning=True,
                config=self.config.get('parser', {}).get('pdf_config', {})
            )
            
            logging.info("PDF parsing completed successfully")
            return parsed_data
            
        except Exception as e:
            logging.error(f"Failed to parse PDF {pdf_path}: {e}")
            raise
    
    def extract_images(self, parsed_data: Dict[str, Any], source_file: str) -> List[Dict[str, Any]]:
        """Extract and filter relevant images from parsed data"""
        images = []
        
        try:
            # Get images from various sources in parsed data
            image_sources = [
                parsed_data.get('images', []),
                parsed_data.get('extracted_images', []),
                parsed_data.get('image_data', [])
            ]
            
            all_images = []
            for source in image_sources:
                if isinstance(source, list):
                    all_images.extend(source)
                elif isinstance(source, dict):
                    all_images.append(source)
            
            logging.info(f"Found {len(all_images)} images to analyze")
            
            relevant_images = []
            for i, image_data in enumerate(all_images):
                try:
                    # Handle different image data formats
                    if isinstance(image_data, str):
                        image_path = image_data
                    elif isinstance(image_data, dict):
                        image_path = image_data.get('path', image_data.get('file_path', ''))
                    else:
                        continue
                    
                    if not os.path.exists(image_path):
                        continue
                    
                    # Check if image is relevant content
                    if self._is_relevant_image(image_path):
                        relevant_images.append((image_path, i + 1))
                        logging.info(f"Relevant image found: {Path(image_path).name}")
                    else:
                        logging.debug(f"Filtered out image: {Path(image_path).name}")
                        
                except Exception as e:
                    logging.warning(f"Failed to analyze image {i+1}: {e}")
                    continue
            
            logging.info(f"Filtered to {len(relevant_images)} relevant images")
            
            # Save relevant images
            for image_path, page_num in relevant_images:
                try:
                    # Generate unique filename
                    original_name = Path(image_path).name
                    filename = self._generate_unique_filename(
                        f"page_{page_num}", 
                        Path(image_path).suffix[1:], 
                        source_file, 
                        'images'
                    )
                    
                    # Copy image to images directory
                    dest_path = self.images_dir / filename
                    import shutil
                    shutil.copy2(image_path, dest_path)
                    
                    # Create image metadata
                    image_info = {
                        'original_path': image_path,
                        'saved_path': str(dest_path),
                        'filename': filename,
                        'page_number': page_num,
                        'file_size': dest_path.stat().st_size,
                        'content_hash': self._calculate_content_hash(str(dest_path))
                    }
                    
                    images.append(image_info)
                    logging.info(f"Saved relevant image: {filename}")
                    
                except Exception as e:
                    logging.warning(f"Failed to save image {page_num}: {e}")
                    continue
            
            logging.info(f"Successfully processed {len(images)} relevant images")
            return images
            
        except Exception as e:
            logging.error(f"Failed to extract images: {e}")
            return []
    
    def extract_tables(self, parsed_data: Dict[str, Any], source_file: str) -> List[Dict[str, Any]]:
        """Extract and clean the main table from parsed data"""
        tables = []
        
        try:
            # Extract the best table
            clean_table = self._extract_clean_table(parsed_data)
            
            if not clean_table:
                logging.info("No suitable table found for extraction")
                return tables
            
            # Generate unique filename
            csv_filename = self._generate_unique_filename(
                "table", 'csv', source_file, 'tables'
            )
            json_filename = self._generate_unique_filename(
                "table", 'json', source_file, 'tables'
            )
            
            # Save as CSV
            csv_path = self.tables_dir / csv_filename
            df = pd.DataFrame(clean_table)
            df.to_csv(csv_path, index=False, encoding='utf-8', header=False)
            
            # Save as JSON with metadata
            json_path = self.tables_dir / json_filename
            table_metadata = {
                'data': clean_table,
                'rows': len(clean_table),
                'columns': len(clean_table[0]) if clean_table else 0,
                'extraction_timestamp': datetime.now().isoformat(),
                'source_file': source_file
            }
            
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(table_metadata, f, indent=2, ensure_ascii=False)
            
            # Create table metadata
            table_info = {
                'table_number': 1,
                'csv_file': str(csv_path),
                'json_file': str(json_path),
                'rows': len(clean_table),
                'columns': len(clean_table[0]) if clean_table else 0,
                'content_hash': self._calculate_content_hash(clean_table)
            }
            
            tables.append(table_info)
            logging.info(f"Saved cleaned table: {csv_filename} ({len(clean_table)} rows)")
            
            return tables
            
        except Exception as e:
            logging.error(f"Failed to extract tables: {e}")
            return []
    
    def extract_text(self, parsed_data: Dict[str, Any], source_file: str) -> Dict[str, Any]:
        """Extract and clean text content from parsed data"""
        try:
            # Collect text from various sources
            text_sources = [
                parsed_data.get('text', ''),
                parsed_data.get('extracted_text', ''),
                parsed_data.get('ocr_text', ''),
                parsed_data.get('handwritten_text', ''),
                parsed_data.get('image_text', ''),
                parsed_data.get('structured_text', '')
            ]
            
            # Combine all text sources
            combined_text = '\n\n'.join([text for text in text_sources if text])
            
            if not combined_text.strip():
                logging.warning("No text content found in document")
                return {}
            
            # Clean the text
            cleaned_text = self._clean_text(combined_text)
            
            # Generate unique filename
            filename = self._generate_unique_filename(
                'extracted_text', 'txt', source_file, 'text'
            )
            
            # Save text content
            text_path = self.text_dir / filename
            with open(text_path, 'w', encoding='utf-8') as f:
                f.write(cleaned_text)
            
            # Create text metadata
            text_info = {
                'filename': filename,
                'file_path': str(text_path),
                'text_length': len(cleaned_text),
                'word_count': len(cleaned_text.split()),
                'line_count': len(cleaned_text.splitlines()),
                'content_hash': self._calculate_content_hash(cleaned_text),
                'extraction_timestamp': datetime.now().isoformat()
            }
            
            logging.info(f"Saved cleaned text content: {filename}")
            return text_info
            
        except Exception as e:
            logging.error(f"Failed to extract text: {e}")
            return {}
    
    def _clean_text(self, text: str) -> str:
        """Clean and format extracted text"""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove common OCR artifacts
        text = re.sub(r'[^\w\s,.\-!?]', '', text)
        
        # Clean up multiple spaces
        text = re.sub(r' +', ' ', text)
        
        # Remove empty lines
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        
        return '\n'.join(lines)
    
    def process_document(self, pdf_path: str) -> Dict[str, Any]:
        """Main method to process a PDF document"""
        start_time = datetime.now()
        
        try:
            logging.info(f"Starting refined document processing: {pdf_path}")
            
            # Validate input file
            if not os.path.exists(pdf_path):
                raise FileNotFoundError(f"PDF file not found: {pdf_path}")
            
            # Parse PDF
            parsed_data = self.parse_pdf(pdf_path)
            
            # Extract content with improved filtering
            images = self.extract_images(parsed_data, pdf_path)
            tables = self.extract_tables(parsed_data, pdf_path)
            text = self.extract_text(parsed_data, pdf_path)
            
            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Create processing summary
            summary = {
                'source_file': pdf_path,
                'processing_timestamp': start_time.isoformat(),
                'processing_time_seconds': processing_time,
                'output_directory': str(self.base_output_dir),
                'content_extracted': {
                    'images': len(images),
                    'tables': len(tables),
                    'text_files': 1 if text else 0
                },
                'images': images,
                'tables': tables,
                'text': text,
                'status': 'success'
            }
            
            # Save processing summary
            summary_filename = f"processing_summary_{Path(pdf_path).stem}.json"
            summary_path = self.base_output_dir / summary_filename
            with open(summary_path, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
            
            logging.info(f"Refined document processing completed successfully in {processing_time:.2f}s")
            logging.info(f"Extracted: {len(images)} relevant images, {len(tables)} clean tables, 1 text file")
            
            return summary
            
        except Exception as e:
            error_msg = f"Failed to process document {pdf_path}: {e}"
            logging.error(error_msg)
            return {
                'source_file': pdf_path,
                'status': 'error',
                'error': error_msg,
                'processing_timestamp': start_time.isoformat()
            }
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get current processing statistics"""
        stats = {
            'output_directory': str(self.base_output_dir),
            'directory_structure': {
                'images': str(self.images_dir),
                'tables': str(self.tables_dir),
                'text': str(self.text_dir)
            },
            'file_counts': {
                'images': len(list(self.images_dir.glob('*'))),
                'tables': len(list(self.tables_dir.glob('*.csv'))),
                'text_files': len(list(self.text_dir.glob('*.txt')))
            }
        }
        return stats


def main():
    """Main function for command-line usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Refined PDF Parsing Pipeline")
    parser.add_argument('--input', '-i', required=True, help='Input PDF file path')
    parser.add_argument('--output', '-o', help='Output directory (default: ./outputs)')
    parser.add_argument('--config', '-c', default='streamlined_config.yaml', help='Configuration file')
    parser.add_argument('--timestamps', action='store_true', help='Include timestamps in filenames')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Setup logging
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Initialize pipeline
    pipeline = RefinedPipeline(args.config)
    
    # Override output directory if provided
    if args.output:
        pipeline.base_output_dir = Path(args.output)
        pipeline.images_dir = pipeline.base_output_dir / 'images'
        pipeline.tables_dir = pipeline.base_output_dir / 'tables'
        pipeline.text_dir = pipeline.base_output_dir / 'text'
        pipeline._create_output_directories()
    
    # Override timestamp setting if provided
    if args.timestamps:
        pipeline.use_timestamps = True
    
    # Process document
    result = pipeline.process_document(args.input)
    
    if result['status'] == 'success':
        print(f"✅ Document processed successfully with refined pipeline!")
        print(f"   Input: {result['source_file']}")
        print(f"   Output: {result['output_directory']}")
        print(f"   Processing time: {result['processing_time_seconds']:.2f}s")
        print(f"   Content extracted: {result['content_extracted']}")
        
        # Show file counts
        stats = pipeline.get_processing_stats()
        print(f"   Files created:")
        print(f"     - Images: {stats['file_counts']['images']}")
        print(f"     - Tables: {stats['file_counts']['tables']}")
        print(f"     - Text files: {stats['file_counts']['text_files']}")
        
        return 0
    else:
        print(f"❌ Processing failed: {result['error']}")
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
