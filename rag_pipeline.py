import os
import json
import math
import argparse
import logging
from typing import List, Dict, Tuple

import yaml
from od_parse import parse_pdf
from PIL import Image
import pytesseract


def setup_logging(enable: bool, log_file: str) -> None:
    if enable:
        logging.basicConfig(filename=log_file, level=logging.INFO,
                            format='%(asctime)s - %(levelname)s - %(message)s')
    else:
        logging.basicConfig(level=logging.INFO,
                            format='%(asctime)s - %(levelname)s - %(message)s')


def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def parse_image(image_path: str) -> str:
    try:
        image = Image.open(image_path)
        return pytesseract.image_to_string(image)
    except Exception as exc:
        logging.error(f"Image parse failed for {image_path}: {exc}")
        return ""


def parse_text_file(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    except Exception as exc:
        logging.error(f"Text parse failed for {path}: {exc}")
        return ""


def extract_text_from_path(path: str) -> str:
    lower = path.lower()
    if lower.endswith('.pdf'):
        try:
            parsed = parse_pdf(path, config={"extract_tables": False})
            return (parsed or {}).get("text", "")
        except Exception as exc:
            logging.error(f"PDF parse failed for {path}: {exc}")
            return ""
    if lower.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')):
        return parse_image(path)
    return parse_text_file(path)


def chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    if not text:
        return []
    tokens = text.split()
    chunks: List[str] = []
    i = 0
    while i < len(tokens):
        chunk_tokens = tokens[i:i + chunk_size]
        if not chunk_tokens:
            break
        chunks.append(" ".join(chunk_tokens))
        if overlap > 0:
            i += max(1, chunk_size - overlap)
        else:
            i += chunk_size
    return chunks


def build_corpus(data_dir: str, chunk_size: int, overlap: int) -> List[Dict[str, str]]:
    documents: List[Dict[str, str]] = []
    for root, _, files in os.walk(data_dir):
        for name in files:
            path = os.path.join(root, name)
            text = extract_text_from_path(path).strip()
            if not text:
                continue
            for idx, chunk in enumerate(chunk_text(text, chunk_size, overlap)):
                documents.append({
                    "source": path,
                    "chunk_index": idx,
                    "text": chunk
                })
    return documents


def tokenize(text: str) -> List[str]:
    out: List[str] = []
    word = []
    for ch in text.lower():
        if ch.isalnum():
            word.append(ch)
        else:
            if word:
                out.append(''.join(word))
                word = []
    if word:
        out.append(''.join(word))
    return out


def compute_tfidf_index(docs: List[Dict[str, str]]) -> Dict:
    # Build vocabulary and document frequencies
    doc_tokens: List[List[str]] = [tokenize(d["text"]) for d in docs]
    df: Dict[str, int] = {}
    for tokens in doc_tokens:
        seen = set(tokens)
        for t in seen:
            df[t] = df.get(t, 0) + 1

    n_docs = max(1, len(docs))
    idf: Dict[str, float] = {t: math.log((n_docs + 1) / (c + 1)) + 1.0 for t, c in df.items()}

    # Compute tf-idf vectors
    vectors: List[Dict[str, float]] = []
    norms: List[float] = []
    for tokens in doc_tokens:
        tf: Dict[str, int] = {}
        for t in tokens:
            tf[t] = tf.get(t, 0) + 1
        vec: Dict[str, float] = {t: (tf[t] / len(tokens)) * idf.get(t, 0.0) for t in tf}
        norm = math.sqrt(sum(v * v for v in vec.values())) or 1.0
        vectors.append(vec)
        norms.append(norm)

    return {
        "documents": docs,
        "idf": idf,
        "vectors": vectors,
        "norms": norms,
    }


def cosine_similarity(query_vec: Dict[str, float], query_norm: float,
                      doc_vec: Dict[str, float], doc_norm: float) -> float:
    if query_norm == 0 or doc_norm == 0:
        return 0.0
    # Iterate over smaller dict for speed
    if len(query_vec) > len(doc_vec):
        query_vec, doc_vec = doc_vec, query_vec
        query_norm, doc_norm = doc_norm, query_norm
    dot = 0.0
    for t, v in query_vec.items():
        if t in doc_vec:
            dot += v * doc_vec[t]
    return dot / (query_norm * doc_norm)


def build_query_vector(text: str, idf: Dict[str, float]) -> Tuple[Dict[str, float], float]:
    tokens = tokenize(text)
    if not tokens:
        return {}, 0.0
    tf: Dict[str, int] = {}
    for t in tokens:
        tf[t] = tf.get(t, 0) + 1
    vec: Dict[str, float] = {t: (tf[t] / len(tokens)) * idf.get(t, 0.0) for t in tf}
    norm = math.sqrt(sum(v * v for v in vec.values())) or 1.0
    return vec, norm


def retrieve(index: Dict, query: str, top_k: int) -> List[Tuple[float, Dict[str, str]]]:
    q_vec, q_norm = build_query_vector(query, index["idf"])
    scored: List[Tuple[float, Dict[str, str]]] = []
    for vec, norm, doc in zip(index["vectors"], index["norms"], index["documents"]):
        score = cosine_similarity(q_vec, q_norm, vec, norm)
        scored.append((score, doc))
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[:top_k]


def save_index(index_path: str, index: Dict) -> None:
    # Convert keys of vectors to compact lists to reduce file size
    serializable = {
        "documents": index["documents"],
        "idf": index["idf"],
        "vectors": index["vectors"],
        "norms": index["norms"],
    }
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(serializable, f)


def load_index(index_path: str) -> Dict:
    with open(index_path, "r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Minimal RAG pipeline using od-parse for parsing")
    parser.add_argument("action", choices=["build", "query"], help="build the index or run a query")
    parser.add_argument("--config", default="config.yaml", help="Path to config file")
    parser.add_argument("--data-dir", default=None, help="Data directory override")
    parser.add_argument("--index-path", default=None, help="Index path override")
    parser.add_argument("--question", default=None, help="Question to query with (for action=query)")
    args = parser.parse_args()

    cfg = load_config(args.config)

    setup_logging(cfg.get('logging', {}).get('enable_logging', True),
                  cfg.get('logging', {}).get('log_file', 'pipeline.log'))

    rag_cfg = cfg.get("rag", {})
    data_dir = args.data_dir or rag_cfg.get("data_dir", "./data")
    index_path = args.index_path or rag_cfg.get("index_path", "./rag_index.json")
    chunk_size = int(rag_cfg.get("chunk_size", 500))
    chunk_overlap = int(rag_cfg.get("chunk_overlap", 50))
    top_k = int(rag_cfg.get("top_k", 3))

    if args.action == "build":
        if not os.path.isdir(data_dir):
            raise SystemExit(f"Data directory not found: {data_dir}")
        logging.info(f"Building corpus from {data_dir} ...")
        docs = build_corpus(data_dir, chunk_size, chunk_overlap)
        logging.info(f"Built {len(docs)} chunks. Computing TF-IDF index ...")
        index = compute_tfidf_index(docs)
        save_index(index_path, index)
        logging.info(f"Index saved to {index_path}")
        return

    if args.action == "query":
        question = args.question or ""
        if not question:
            raise SystemExit("Provide --question for action=query")
        if not os.path.isfile(index_path):
            raise SystemExit(f"Index not found at {index_path}. Run build first.")
        index = load_index(index_path)
        hits = retrieve(index, question, top_k)
        print("Top results:\n")
        for rank, (score, doc) in enumerate(hits, 1):
            print(f"{rank}. score={score:.4f} source={doc['source']} chunk={doc['chunk_index']}")
            print(doc['text'][:400] + ("..." if len(doc['text']) > 400 else ""))
            print()


if __name__ == "__main__":
    main()



