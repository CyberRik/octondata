import yaml
import os
import logging
from od_parse import parse_pdf
import pytesseract
from PIL import Image

# Set up logging (if enabled in the config)
def setup_logging(log_file):
    logging.basicConfig(filename=log_file, level=logging.INFO, 
                        format='%(asctime)s - %(levelname)s - %(message)s')

# Load configuration from config.yaml
def load_config(config_path="config.yaml"):
    with open(config_path, "r") as file:
        return yaml.safe_load(file)

# Parse the document based on file type
def parse_document(file_path, parser_config):
    try:
        if parser_config['pdf']:
            # Parse PDF document using od-parse with table extraction enabled
            parsed_data = parse_pdf(file_path, config={"extract_tables": True})
            document_text = parsed_data.get("text", "")
            tables = parsed_data.get("tables", [])
            logging.info(f"Extracted {len(tables)} tables from the document.")
            return document_text, tables
        elif parser_config['image']:
            # Implement image parsing using Tesseract OCR
            document_text = parse_image(file_path)
            logging.info("Image parsed successfully.")
            return document_text, []
        elif parser_config['text']:
            # Implement text file parsing
            document_text = parse_text(file_path)
            logging.info("Text file parsed successfully.")
            return document_text, []
        else:
            raise ValueError("No valid parser type specified.")
    except Exception as e:
        logging.error(f"Error parsing document {file_path}: {e}")
        return None, None

# Generate text using a Hugging Face model
def generate_text(prompt, model_name, max_length, temperature):
    # Deprecated: generation removed to avoid duplicate/unnecessary outputs
    return None
    
# Image Parsing using Tesseract OCR
def parse_image(image_path):
    try:
        # Open the image using Pillow
        image = Image.open(image_path)
        # Use Tesseract OCR to extract text
        text = pytesseract.image_to_string(image)
        return text
    except Exception as e:
        logging.error(f"Error parsing image {image_path}: {e}")
        return ""

# Text Parsing for plain text files
def parse_text(text_file_path):
    try:
        with open(text_file_path, 'r') as file:
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
        document_text, tables = parse_document(input_file, config['parser'])
        if document_text is None:
            logging.error("Document parsing failed.")
            return
        logging.info("Document parsed successfully.")
    else:
        logging.error(f"Input file {input_file} does not exist.")
        return

    # Save full parsed document text only
    os.makedirs(output_dir, exist_ok=True)
    parsed_text_file = os.path.join(output_dir, "parsed_text.txt")
    try:
        with open(parsed_text_file, 'w', encoding='utf-8', newline='') as ftxt:
            ftxt.write(document_text or "")
        logging.info(f"Parsed text saved to {parsed_text_file}")
    except Exception as e:
        logging.error(f"Failed to write parsed text file: {e}")

    # If tables were extracted, you can also save the tables (optional)
    if tables:
        tables_output_file = os.path.join(output_dir, "extracted_tables.md")
        with open(tables_output_file, 'w', encoding='utf-8', newline='') as file:
            for i, table in enumerate(tables):
                file.write(f"Table {i + 1}:\n")
                file.write(table.to_markdown() + "\n")
        logging.info(f"Extracted tables saved to {tables_output_file}")

# Run the pipeline
if __name__ == "__main__":
    config = load_config()  # Load the config from the YAML file
    run_pipeline(config)
