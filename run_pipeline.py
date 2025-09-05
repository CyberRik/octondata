import yaml
import os
import logging
from od_parse import parse_pdf

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
            # Parse PDF via od-parse; enable tables and internal OCR if available
            parsed_data = parse_pdf(file_path, config={"extract_tables": True, "ocr": True})
            document_text = parsed_data.get("text", "") or ""
            # Append any od-parse OCR outputs, if exposed in result
            ocr_candidates_keys = [
                "ocr_text",            # string or list of strings
                "image_texts",         # list of strings
                "ocr",                 # dict with 'text' or list
                "images_ocr_text",     # alt key
            ]
            ocr_segments = []
            for key in ocr_candidates_keys:
                if key in parsed_data and parsed_data[key]:
                    val = parsed_data[key]
                    if isinstance(val, str):
                        ocr_segments.append(val)
                    elif isinstance(val, (list, tuple)):
                        for it in val:
                            if isinstance(it, str) and it.strip():
                                ocr_segments.append(it)
                    elif isinstance(val, dict):
                        txt = val.get("text") if isinstance(val.get("text"), str) else None
                        if txt:
                            ocr_segments.append(txt)
            if ocr_segments:
                if document_text.strip():
                    document_text += "\n\n"
                document_text += "\n\n".join(s.strip() for s in ocr_segments if s.strip())
                logging.info("Appended OCR text from od-parse outputs to parsed text.")
            tables = parsed_data.get("tables", [])
            logging.info(f"Extracted {len(tables)} tables from the document.")
            return document_text, tables
        elif parser_config['image']:
            # Delegate image parsing to od-parse by routing through parse_pdf-like interface if supported
            # For simplicity, treat as text-only path (od-parse handles images internally when passed)
            parsed_data = parse_pdf(file_path, config={"ocr": True, "extract_tables": False})
            document_text = parsed_data.get("text", "") or ""
            # Append any OCR text keys as above
            ocr_segments = []
            for key in ("ocr_text", "image_texts", "ocr", "images_ocr_text"):
                if key in parsed_data and parsed_data[key]:
                    val = parsed_data[key]
                    if isinstance(val, str):
                        ocr_segments.append(val)
                    elif isinstance(val, (list, tuple)):
                        ocr_segments.extend([s for s in val if isinstance(s, str)])
                    elif isinstance(val, dict) and isinstance(val.get("text"), str):
                        ocr_segments.append(val["text"])
            if ocr_segments:
                if document_text.strip():
                    document_text += "\n\n"
                document_text += "\n\n".join(s.strip() for s in ocr_segments if isinstance(s, str))
            logging.info("Image parsed successfully via od-parse.")
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

# Generate text (deprecated)
def generate_text(prompt, model_name, max_length, temperature):
    return None

# Text Parsing for plain text files
def parse_text(text_file_path):
    try:
        with open(text_file_path, 'r') as file:
            text = file.read()
        return text
    except Exception as e:
        logging.error(f"Error parsing text file {text_file_path}: {e}")
        return ""

def _render_table_markdown(table):
    try:
        # Pandas DataFrame
        to_md = getattr(table, 'to_markdown', None)
        if callable(to_md):
            return to_md()
        # List of lists or list of dicts
        if isinstance(table, list) and table:
            # If dicts, try to normalize special pattern with 'Unnamed: N' columns
            if isinstance(table[0], dict):
                first = table[0]
                keys = list(first.keys())
                unnamed = [(k, int(k.split(':')[1].strip())) for k in keys if k.startswith('Unnamed:') and k.split(':')[1].strip().isdigit()]
                # primary key: pick the longest key string as the first column (often paragraph-like header)
                other_keys = [k for k in keys if not k.startswith('Unnamed:')]
                primary_key = max(other_keys, key=lambda k: len(str(k)), default=(other_keys[0] if other_keys else None))
                if primary_key or unnamed:
                    ordered_cols = []
                    if primary_key:
                        ordered_cols.append(primary_key)
                    ordered_cols.extend([k for k, _n in sorted(unnamed, key=lambda x: x[1])])
                    # Build header from first row values
                    header_vals = [str(first.get(col, '')) for col in ordered_cols]
                    body_rows = []
                    for row in table[1:]:
                        body_rows.append([str(row.get(col, '')) for col in ordered_cols])
                    md = "| " + " | ".join(header_vals) + " |\n"
                    md += "| " + " | ".join(["---"] * len(header_vals)) + " |\n"
                    for r in body_rows:
                        md += "| " + " | ".join(r) + " |\n"
                    return md
                # Fallback: use keys order as-is
                headers = list(first.keys())
                rows = [[str(row.get(h, '')) for h in headers] for row in table]
                md = "| " + " | ".join(headers) + " |\n"
                md += "| " + " | ".join(["---"] * len(headers)) + " |\n"
                for r in rows:
                    md += "| " + " | ".join(r) + " |\n"
                return md
            else:
                # assume first row is header if elements are scalar
                headers = [str(h) for h in table[0]]
                rows = [[str(c) for c in r] for r in table[1:]]
                md = "| " + " | ".join(headers) + " |\n"
                md += "| " + " | ".join(["---"] * len(headers)) + " |\n"
                for r in rows:
                    md += "| " + " | ".join(r) + " |\n"
                return md
        # Dict with values list-like
        if isinstance(table, dict) and table:
            headers = list(table.keys())
            # transpose values if lists of equal length
            values = list(table.values())
            row_count = max((len(v) for v in values if isinstance(v, (list, tuple))), default=1)
            rows = []
            for i in range(row_count):
                row = []
                for v in values:
                    if isinstance(v, (list, tuple)) and i < len(v):
                        row.append(str(v[i]))
                    else:
                        row.append(str(v) if i == 0 else '')
                rows.append(row)
            md = "| " + " | ".join(headers) + " |\n"
            md += "| " + " | ".join(["---"] * len(headers)) + " |\n"
            for r in rows:
                md += "| " + " | ".join(r) + " |\n"
            return md
    except Exception as exc:
        logging.warning(f"Failed to render table to markdown: {exc}")
    return None

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
    # No longer writing extracted tables to file; focus on parsed text + OCR.

# Run the pipeline
if __name__ == "__main__":
    config = load_config()  # Load the config from the YAML file
    run_pipeline(config)
