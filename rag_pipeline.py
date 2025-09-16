#!/usr/bin/env python3
"""
RAG (Retrieve and Generate) Pipeline
Integrates with the refined PDF parsing pipeline to provide retrieval and generation capabilities
"""

import os
import json
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Any
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import yaml

# Import the refined pipeline
from enhanced_refined_pipeline import EnhancedRefinedPipeline


def setup_logging(enable_logging: bool = True, log_file: str = 'rag_pipeline.log'):
    """Setup logging configuration"""
    if enable_logging:
        logging.basicConfig(
            filename=log_file,
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    else:
        logging.basicConfig(level=logging.INFO)


def load_config(config_path: str = "streamlined_config.yaml") -> Dict:
    """Load configuration from YAML file"""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            return config
    except Exception as e:
        logging.error(f"Failed to load config: {e}")
        return {}


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """Split text into overlapping chunks for better retrieval"""
    if not text:
        return []
    
    words = text.split()
    chunks = []
    
    for i in range(0, len(words), chunk_size - overlap):
        chunk = ' '.join(words[i:i + chunk_size])
        if chunk.strip():
            chunks.append(chunk.strip())
    
    return chunks


def build_corpus_from_parsed_data(parsed_data: Dict[str, Any]) -> List[Dict[str, str]]:
    """Build corpus from parsed PDF data"""
    documents = []
    
    # Add text content from the enhanced pipeline output
    if 'text' in parsed_data and parsed_data['text']:
        text_content = parsed_data['text'].get('text_content', '')
        if not text_content and 'file_path' in parsed_data['text']:
            # Try to read from file if text_content is not available
            try:
                with open(parsed_data['text']['file_path'], 'r', encoding='utf-8') as f:
                    text_content = f.read()
            except:
                pass
        
        if text_content:
            chunks = chunk_text(text_content)
            for i, chunk in enumerate(chunks):
                documents.append({
                    "source": "text",
                    "chunk_index": i,
                    "text": chunk,
                    "type": "text_content"
                })
    
    # Add table content from the enhanced pipeline output
    if 'tables' in parsed_data and parsed_data['tables']:
        for table_idx, table in enumerate(parsed_data['tables']):
            # Try to read table data from CSV file
            if 'csv_file' in table:
                try:
                    import pandas as pd
                    df = pd.read_csv(table['csv_file'], header=None)
                    table_text = ""
                    for _, row in df.iterrows():
                        table_text += " | ".join(str(cell) for cell in row) + "\n"
                    
                    if table_text.strip():
                        documents.append({
                            "source": f"table_{table_idx + 1}",
                            "chunk_index": 0,
                            "text": table_text.strip(),
                            "type": "table_data"
                        })
                except Exception as e:
                    logging.warning(f"Failed to read table CSV: {e}")
            
            # Fallback to data field if available
            elif 'data' in table:
                table_text = ""
                for row in table['data']:
                    if isinstance(row, list):
                        table_text += " | ".join(str(cell) for cell in row) + "\n"
                    elif isinstance(row, dict):
                        table_text += " | ".join(f"{k}: {v}" for k, v in row.items()) + "\n"
                
                if table_text.strip():
                    documents.append({
                        "source": f"table_{table_idx + 1}",
                        "chunk_index": 0,
                        "text": table_text.strip(),
                        "type": "table_data"
                    })
    
    # Add image metadata
    if 'images' in parsed_data and parsed_data['images']:
        for img_idx, image in enumerate(parsed_data['images']):
            # Extract any OCR text from images
            if 'ocr_text' in image:
                ocr_text = image['ocr_text']
                if ocr_text and ocr_text.strip():
                    documents.append({
                        "source": f"image_{img_idx + 1}",
                        "chunk_index": 0,
                        "text": ocr_text.strip(),
                        "type": "image_ocr"
                    })
            
            # Add image filename as metadata
            if 'filename' in image:
                documents.append({
                    "source": f"image_{img_idx + 1}",
                    "chunk_index": 0,
                    "text": f"Image: {image['filename']} (embedded: {image.get('is_embedded', False)})",
                    "type": "image_metadata"
                })
    
    logging.info(f"Built corpus with {len(documents)} documents from parsed data")
    return documents


def compute_tfidf_index(documents: List[Dict[str, str]]) -> Dict[str, Any]:
    """Compute TF-IDF index for document retrieval"""
    if not documents:
        return {"documents": [], "tfidf_matrix": None, "vectorizer": None}
    
    # Extract text content
    texts = [doc["text"] for doc in documents]
    
    # Create TF-IDF vectorizer
    vectorizer = TfidfVectorizer(
        max_features=1000,
        stop_words='english',
        ngram_range=(1, 2),
        min_df=1,
        max_df=0.95
    )
    
    # Fit and transform texts
    tfidf_matrix = vectorizer.fit_transform(texts)
    
    # Create index
    index = {
        "documents": documents,
        "tfidf_matrix": tfidf_matrix.toarray().tolist(),
        "vectorizer": {
            "vocabulary": vectorizer.vocabulary_,
            "idf": vectorizer.idf_.tolist(),
            "stop_words": list(vectorizer.stop_words_) if hasattr(vectorizer, 'stop_words_') else []
        },
        "feature_names": vectorizer.get_feature_names_out().tolist()
    }
    
    logging.info(f"Computed TF-IDF index with {len(documents)} documents")
    return index


def save_index(index: Dict[str, Any], index_path: str):
    """Save TF-IDF index to file"""
    try:
        with open(index_path, 'w', encoding='utf-8') as f:
            json.dump(index, f, indent=2, ensure_ascii=False, default=str)
        logging.info(f"Index saved to {index_path}")
    except Exception as e:
        logging.error(f"Failed to save index: {e}")


def load_index(index_path: str) -> Dict[str, Any]:
    """Load TF-IDF index from file"""
    try:
        with open(index_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logging.error(f"Failed to load index: {e}")
        return {"documents": [], "tfidf_matrix": None, "vectorizer": None}


def retrieve_documents(query: str, index: Dict[str, Any], top_k: int = 3) -> List[Dict[str, Any]]:
    """Retrieve most relevant documents for a query"""
    if not index.get("documents") or not index.get("tfidf_matrix"):
        return []
    
    try:
        # Get documents and create fresh vectorizer (avoid vocabulary reconstruction issues)
        documents = index["documents"]
        texts = [doc["text"] for doc in documents]
        
        # Create fresh vectorizer
        vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=1,
            max_df=0.95
        )
        
        # Fit on texts and transform query
        tfidf_matrix = vectorizer.fit_transform(texts)
        query_vector = vectorizer.transform([query])
        
        # Compute similarities
        similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()
        
        # Get top-k most similar documents
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            if similarities[idx] > 0:  # Only include documents with similarity > 0
                doc = documents[idx].copy()
                doc["similarity_score"] = float(similarities[idx])
                results.append(doc)
        
        logging.info(f"Retrieved {len(results)} documents for query: {query[:50]}...")
        return results
        
    except Exception as e:
        logging.error(f"Failed to retrieve documents: {e}")
        return []


def generate_response(query: str, retrieved_docs: List[Dict[str, Any]]) -> str:
    """Generate a response based on retrieved documents"""
    if not retrieved_docs:
        return "No relevant information found for your query."
    
    # Combine retrieved documents
    context = "\n\n".join([
        f"[{doc['type']}] {doc['text']}" 
        for doc in retrieved_docs
    ])
    
    # Simple response generation (can be enhanced with LLM integration)
    response = f"""Based on the retrieved information:

Query: {query}

Relevant Information:
{context}

Summary: The retrieved documents contain information related to your query. The most relevant content has been extracted and presented above."""
    
    return response


class RAGPipeline:
    """RAG Pipeline that integrates with the enhanced refined PDF parsing pipeline"""
    
    def __init__(self, config_path: str = "streamlined_config.yaml"):
        self.config = load_config(config_path)
        self.parser = EnhancedRefinedPipeline(config_path)
        self.index = None
        self.index_path = self.config.get('rag', {}).get('index_path', './rag_index.json')
        
        # RAG configuration
        self.rag_config = self.config.get('rag', {})
        self.chunk_size = self.rag_config.get('chunk_size', 500)
        self.chunk_overlap = self.rag_config.get('chunk_overlap', 50)
        self.top_k = self.rag_config.get('top_k', 3)
        
        logging.info("RAG Pipeline initialized successfully")
    
    def process_document_and_build_index(self, pdf_path: str, index_path: str = None) -> Dict[str, Any]:
        """Process PDF and build RAG index"""
        try:
            logging.info(f"Processing document and building RAG index: {pdf_path}")
            
            # Process PDF with enhanced pipeline
            parsed_result = self.parser.process_document(pdf_path)
            
            if parsed_result['status'] != 'success':
                return {
                    'status': 'error',
                    'error': f"PDF processing failed: {parsed_result.get('error', 'Unknown error')}"
                }
            
            # Build corpus from parsed data
            documents = build_corpus_from_parsed_data(parsed_result)
            
            if not documents:
                return {
                    'status': 'error',
                    'error': "No documents found in parsed data"
                }
            
            # Compute TF-IDF index
            index = compute_tfidf_index(documents)
            
            # Save index
            index_path = index_path or self.index_path
            save_index(index, index_path)
            
            # Store index in memory
            self.index = index
            
            # Clean up temporary files
            self._cleanup_temp_files()

            return {
                'status': 'success',
                'parsed_data': parsed_result,
                'documents_count': len(documents),
                'index_path': index_path
            }
            
        except Exception as e:
            error_msg = f"Failed to process document and build index: {e}"
            logging.error(error_msg)
            return {'status': 'error', 'error': error_msg}
    
    def load_index(self, index_path: str = None) -> bool:
        """Load existing RAG index"""
        try:
            index_path = index_path or self.index_path
            self.index = load_index(index_path)
            
            if not self.index.get("documents"):
                logging.warning("Loaded index is empty")
                return False
            
            logging.info(f"Loaded RAG index with {len(self.index['documents'])} documents")
            return True
            
        except Exception as e:
            logging.error(f"Failed to load index: {e}")
            return False
    
    def query(self, question: str, top_k: int = None) -> Dict[str, Any]:
        """Query the RAG system"""
        if not self.index:
            return {
                'status': 'error',
                'error': 'No index loaded. Please build or load an index first.'
            }
        
        try:
            top_k = top_k or self.top_k
            
            # Retrieve relevant documents
            retrieved_docs = retrieve_documents(question, self.index, top_k)
            
            if not retrieved_docs:
                return {
                    'status': 'success',
                    'question': question,
                    'answer': "No relevant information found for your query.",
                    'retrieved_documents': [],
                    'confidence': 0.0
                }
            
            # Generate response
            answer = generate_response(question, retrieved_docs)
            
            # Calculate average confidence
            avg_confidence = np.mean([doc['similarity_score'] for doc in retrieved_docs])
            
            return {
                'status': 'success',
                'question': question,
                'answer': answer,
                'retrieved_documents': retrieved_docs,
                'confidence': float(avg_confidence)
            }
            
        except Exception as e:
            error_msg = f"Failed to process query: {e}"
            logging.error(error_msg)
            return {'status': 'error', 'error': error_msg}
    
    def get_index_stats(self) -> Dict[str, Any]:
        """Get statistics about the current index"""
        if not self.index:
            return {'status': 'error', 'error': 'No index loaded'}
        
        documents = self.index.get('documents', [])
        
        # Count by type
        type_counts = {}
        for doc in documents:
            doc_type = doc.get('type', 'unknown')
            type_counts[doc_type] = type_counts.get(doc_type, 0) + 1
        
        return {
            'status': 'success',
            'total_documents': len(documents),
            'type_distribution': type_counts,
            'index_path': self.index_path
        }
    
    def _cleanup_temp_files(self):
        """Clean up temporary files created during processing"""
        try:
            from pathlib import Path
            import shutil
            
            # Clean up sample_images directory (created by od-parse)
            sample_images_dir = Path("sample_images")
            if sample_images_dir.exists():
                shutil.rmtree(sample_images_dir)
                logging.info("Cleaned up temporary sample_images directory")
            
            # Clean up default outputs directory if it's empty
            default_outputs = Path("outputs")
            if default_outputs.exists():
                # Check if it's empty
                is_empty = True
                for subdir in ["images", "tables", "text"]:
                    subdir_path = default_outputs / subdir
                    if subdir_path.exists() and list(subdir_path.glob("*")):
                        is_empty = False
                        break
                
                if is_empty:
                    shutil.rmtree(default_outputs)
                    logging.info("Cleaned up empty outputs directory")
            
        except Exception as e:
            logging.warning(f"Failed to clean up temporary files: {e}")


def main():
    """Main function for command-line usage"""
    parser = argparse.ArgumentParser(description="RAG Pipeline for PDF Processing")
    parser.add_argument("action", choices=["build", "query", "stats"], 
                       help="Action to perform: build index, query, or get stats")
    parser.add_argument("--pdf", help="PDF file path (for build action)")
    parser.add_argument("--question", help="Question to ask (for query action)")
    parser.add_argument("--index", help="Index file path")
    parser.add_argument("--config", default="streamlined_config.yaml", help="Configuration file")
    parser.add_argument("--top-k", type=int, default=3, help="Number of top documents to retrieve")
    
    args = parser.parse_args()

    # Setup logging
    setup_logging()
    
    # Initialize RAG pipeline
    rag_pipeline = RAGPipeline(args.config)

    if args.action == "build":
        if not args.pdf:
            print("Error: PDF file path required for build action")
            return 1
        
        result = rag_pipeline.process_document_and_build_index(args.pdf, args.index)
        
        if result['status'] == 'success':
            print(f"✅ RAG index built successfully!")
            print(f"   PDF: {args.pdf}")
            print(f"   Documents: {result['documents_count']}")
            print(f"   Index: {result['index_path']}")
        else:
            print(f"❌ Failed to build index: {result['error']}")
            return 1
    
    elif args.action == "query":
        if not args.question:
            print("Error: Question required for query action")
            return 1
        
        # Load index if not already loaded
        if not rag_pipeline.index:
            if not rag_pipeline.load_index(args.index):
                print("Error: No index available. Please build an index first.")
                return 1
        
        result = rag_pipeline.query(args.question, args.top_k)
        
        if result['status'] == 'success':
            print(f"✅ Query processed successfully!")
            print(f"   Question: {result['question']}")
            print(f"   Answer: {result['answer']}")
            print(f"   Confidence: {result['confidence']:.3f}")
            print(f"   Retrieved documents: {len(result['retrieved_documents'])}")
        else:
            print(f"❌ Query failed: {result['error']}")
            return 1
    
    elif args.action == "stats":
        stats = rag_pipeline.get_index_stats()
        
        if stats['status'] == 'success':
            print(f"✅ Index statistics:")
            print(f"   Total documents: {stats['total_documents']}")
            print(f"   Type distribution: {stats['type_distribution']}")
            print(f"   Index path: {stats['index_path']}")
        else:
            print(f"❌ Failed to get stats: {stats['error']}")
            return 1
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
