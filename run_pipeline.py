import yaml
import os
import logging
from od_parse import parse_pdf
import json
from pathlib import Path

# Set up logging (if enabled in the config)
def setup_logging(log_file):
    logging.basicConfig(filename=log_file, level=logging.INFO, 
                        format='%(asctime)s - %(levelname)s - %(message)s')

# Load configuration from config.yaml
def load_config(config_path="config.yaml"):
    with open(config_path, "r") as file:
        return yaml.safe_load(file)

# Parse the document based on file type using advanced od-parse features
def parse_document(file_path, parser_config):
    try:
        if parser_config['pdf']:
            # Use advanced od-parse features for comprehensive parsing
            logging.info("Starting advanced PDF parsing with od-parse...")
            
            # Parse with full pipeline including tables, forms, structure, and OCR
            parsed_data = parse_pdf(
                file_path=file_path,
                output_format="json",
                pipeline_type="full",
                use_deep_learning=True,
                config={
                    "extract_tables": True,
                    "extract_forms": True,
                    "extract_structure": True,
                    "ocr": True,
                    "handwritten_detection": True,
                    "img2table": True,  # Use img2table for better table extraction
                    "image_naming": "descriptive"  # Use descriptive names
                }
            )
            
            # Extract main text content
            document_text = parsed_data.get("text", "") or ""
            
            # Extract and append OCR content from various sources
            ocr_sources = [
                parsed_data.get("ocr_text", ""),
                parsed_data.get("handwritten_text", ""),
                parsed_data.get("image_text", ""),
                parsed_data.get("extracted_text", "")
            ]
            
            # Also check for OCR in nested structures
            if "ocr_results" in parsed_data:
                ocr_results = parsed_data["ocr_results"]
                if isinstance(ocr_results, list):
                    for result in ocr_results:
                        if isinstance(result, dict) and "text" in result:
                            ocr_sources.append(result["text"])
                elif isinstance(ocr_results, dict) and "text" in ocr_results:
                    ocr_sources.append(ocr_results["text"])
            
            # Combine all OCR text
            ocr_text = "\n\n".join([text for text in ocr_sources if text and text.strip()])
            if ocr_text:
                if document_text.strip():
                    document_text += "\n\n=== OCR EXTRACTED CONTENT ===\n\n"
                document_text += ocr_text
                logging.info("Appended OCR content to parsed text.")
            
            # Extract tables information
            tables = parsed_data.get("tables", [])
            if tables:
                table_text = "\n\n=== EXTRACTED TABLES ===\n\n"
                for i, table in enumerate(tables):
                    table_text += f"Table {i+1}:\n"
                    if "markdown" in table:
                        table_text += table["markdown"] + "\n\n"
                    elif "data" in table:
                        table_text += str(table["data"]) + "\n\n"
                    else:
                        table_text += str(table) + "\n\n"
                document_text += table_text
                logging.info(f"Appended {len(tables)} tables to parsed text.")
            
            # Extract forms information
            forms = parsed_data.get("forms", [])
            if forms:
                form_text = "\n\n=== EXTRACTED FORMS ===\n\n"
                for i, form in enumerate(forms):
                    form_text += f"Form Field {i+1}:\n"
                    form_text += f"Type: {form.get('type', 'unknown')}\n"
                    form_text += f"Label: {form.get('label', 'N/A')}\n"
                    form_text += f"Value: {form.get('value', 'N/A')}\n"
                    form_text += f"Page: {form.get('page_number', 'N/A')}\n\n"
                document_text += form_text
                logging.info(f"Appended {len(forms)} form fields to parsed text.")
            
            # Extract structure information
            structure = parsed_data.get("structure", {})
            if structure:
                structure_text = "\n\n=== DOCUMENT STRUCTURE ===\n\n"
                elements = structure.get("elements", [])
                for element in elements:
                    elem_type = element.get("type", "unknown")
                    text = element.get("text", "")
                    if elem_type == "heading":
                        level = element.get("level", 1)
                        structure_text += f"# {'#' * (level-1)} {text}\n\n"
                    elif elem_type == "paragraph":
                        structure_text += f"{text}\n\n"
                    elif elem_type == "list_item":
                        structure_text += f"- {text}\n"
                document_text += structure_text
                logging.info(f"Appended document structure to parsed text.")
            
            logging.info(f"Advanced parsing completed. Extracted {len(tables)} tables, {len(forms)} forms.")
            return document_text, tables, parsed_data
            
        elif parser_config['image']:
            # Parse image using od-parse with advanced OCR
            parsed_data = parse_pdf(
                file_path=file_path,
                pipeline_type="ocr",
                use_deep_learning=True,
                config={"ocr": True, "handwritten_detection": True}
            )
            document_text = parsed_data.get("text", "") or ""
            ocr_text = parsed_data.get("ocr_text", "") or parsed_data.get("handwritten_text", "")
            if ocr_text and ocr_text.strip():
                if document_text.strip():
                    document_text += "\n\n=== OCR CONTENT ===\n\n"
                document_text += ocr_text
            logging.info("Image parsed successfully with advanced OCR.")
            return document_text, [], parsed_data
            
        elif parser_config['text']:
            # Implement text file parsing
            document_text = parse_text(file_path)
            logging.info("Text file parsed successfully.")
            return document_text, [], {}
        else:
            raise ValueError("No valid parser type specified.")
    except Exception as e:
        logging.error(f"Error parsing document {file_path}: {e}")
        return None, None, {}

# Text Parsing for plain text files
def parse_text(text_file_path):
    try:
        with open(text_file_path, 'r', encoding='utf-8') as file:
            text = file.read()
        return text
    except Exception as e:
        logging.error(f"Error parsing text file {text_file_path}: {e}")
        return ""

# Main pipeline function
def run_pipeline(config):
    # Set up logging if enabled
    if config['logging']['enable_logging']:
        setup_logging(config['logging']['log_file'])
        logging.info("Pipeline started.")

    # Load input and output directories
    input_file = config['input_file']
    output_dir = config['output_dir']

    # Parse the document
    if os.path.exists(input_file):
        document_text, tables, parsed_data = parse_document(input_file, config['parser'])
        if document_text is None:
            logging.error("Document parsing failed.")
            return
        logging.info("Document parsed successfully.")
    else:
        logging.error(f"Input file {input_file} does not exist.")
        return

    # Save comprehensive parsed document text with organized file structure
    os.makedirs(output_dir, exist_ok=True)
    
    # Create subdirectories for different content types
    text_dir = os.path.join(output_dir, "text")
    tables_dir = os.path.join(output_dir, "tables")
    images_dir = os.path.join(output_dir, "images")
    
    os.makedirs(text_dir, exist_ok=True)
    os.makedirs(tables_dir, exist_ok=True)
    os.makedirs(images_dir, exist_ok=True)
    
    try:
        # Save main text content
        text_file = os.path.join(text_dir, "extracted_text.txt")
        with open(text_file, 'w', encoding='utf-8', newline='') as ftxt:
            ftxt.write(document_text or "")
        logging.info(f"Text content saved to {text_file}")
        
        # Save tables separately with clean CSV format
        if tables:
            # Remove duplicate tables by comparing their data
            unique_tables = []
            seen_data = set()
            
            for table in tables:
                if isinstance(table, dict) and 'data' in table:
                    # Create a hash of the table data to detect duplicates
                    table_data_str = str(table['data'])
                    if table_data_str not in seen_data:
                        seen_data.add(table_data_str)
                        unique_tables.append(table)
                else:
                    # For non-dict tables, add them as-is
                    unique_tables.append(table)
            
            # Save only unique tables
            for i, table in enumerate(unique_tables):
                table_file = os.path.join(tables_dir, f"table_{i+1}.csv")
                try:
                    if isinstance(table, dict) and 'data' in table:
                        # Clean up the table data - extract only the actual data rows
                        table_data = table['data']
                        if isinstance(table_data, list) and len(table_data) > 0:
                            # Find the actual data rows (skip the messy header rows)
                            clean_rows = []
                            for row in table_data:
                                if isinstance(row, dict):
                                    # Extract values and filter out empty/header rows
                                    values = list(row.values())
                                    # Check if this looks like a data row (not a header with long text)
                                    if len(values) > 0 and not any(len(str(v)) > 50 for v in values):
                                        clean_rows.append(values)
                            
                            if clean_rows:
                                # Create clean CSV with proper headers
                                import pandas as pd
                                # Use simple column names
                                headers = [f"Column_{j+1}" for j in range(len(clean_rows[0]))]
                                df = pd.DataFrame(clean_rows, columns=headers)
                                df.to_csv(table_file, index=False)
                                logging.info(f"Clean table {i+1} saved to {table_file}")
                            else:
                                # Fallback: save as JSON
                                table_file = table_file.replace('.csv', '.json')
                                with open(table_file, 'w', encoding='utf-8') as f:
                                    json.dump(table, f, indent=2, ensure_ascii=False)
                                logging.info(f"Table {i+1} saved as JSON to {table_file}")
                        else:
                            # Fallback: save as JSON
                            table_file = table_file.replace('.csv', '.json')
                            with open(table_file, 'w', encoding='utf-8') as f:
                                json.dump(table, f, indent=2, ensure_ascii=False)
                            logging.info(f"Table {i+1} saved as JSON to {table_file}")
                    else:
                        # For non-dict tables, save as JSON
                        table_file = table_file.replace('.csv', '.json')
                        with open(table_file, 'w', encoding='utf-8') as f:
                            json.dump(table, f, indent=2, ensure_ascii=False)
                        logging.info(f"Table {i+1} saved as JSON to {table_file}")
                except Exception as e:
                    logging.warning(f"Could not save table {i+1}: {e}")
        
        # Save and rename images with descriptive names
        if 'images' in parsed_data and parsed_data['images']:
            image_paths = parsed_data['images']
            logging.info(f"Found {len(image_paths)} images to process")
            
            # Process each image and rename based on content
            processed_count = 0
            for i, img_path in enumerate(image_paths):
                source_path = Path(img_path)
                if source_path.exists():
                    try:
                        # Determine image type and create descriptive name
                        if 'img_' in str(source_path).lower():
                            # This is a cropped image - likely table content
                            dest_name = f"table_crop_{i+1}{source_path.suffix}"
                            dest_path = os.path.join(images_dir, dest_name)
                        elif 'page_' in str(source_path).lower() and 'img_' not in str(source_path).lower():
                            # This is a full page image
                            dest_name = f"full_page{source_path.suffix}"
                            dest_path = os.path.join(images_dir, dest_name)
                        else:
                            # Other images
                            dest_name = f"content_{i+1}{source_path.suffix}"
                            dest_path = os.path.join(images_dir, dest_name)
                        
                        # Copy the image with new name
                        import shutil
                        shutil.copy2(str(source_path), dest_path)
                        logging.info(f"Image saved as {dest_name}")
                        processed_count += 1
                        
                    except Exception as e:
                        logging.warning(f"Could not save image {img_path}: {e}")
                else:
                    logging.warning(f"Image file does not exist: {source_path}")
            
            logging.info(f"Image processing completed. Processed {processed_count} images")
        
        # Save comprehensive JSON output
        json_file = os.path.join(output_dir, "complete_data.json")
        try:
            # Re-parse to get full structured data
            full_data = parse_pdf(
                input_file,
                output_format="json",
                pipeline_type="full",
                use_deep_learning=True,
                config={
                    "extract_tables": True,
                    "extract_forms": True,
                    "extract_structure": True,
                    "ocr": True,
                    "handwritten_detection": True,
                    "img2table": True
                }
            )
            with open(json_file, 'w', encoding='utf-8') as fjson:
                json.dump(full_data, fjson, indent=2, ensure_ascii=False)
            logging.info(f"Complete structured data saved to {json_file}")
        except Exception as e:
            logging.warning(f"Could not save JSON output: {e}")
            
    except Exception as e:
        logging.error(f"Failed to write output files: {e}")

# Run the pipeline
if __name__ == "__main__":
    config = load_config()  # Load the config from the YAML file
    run_pipeline(config)
