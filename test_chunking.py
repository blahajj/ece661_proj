from pypdf import PdfReader
from sklearn.cluster import AgglomerativeClustering, DBSCAN
import numpy as np
from transformers import pipeline
from chunking import chunk_data  # Import your chunking function if in a separate module

def preprocess_data(file_path):
    pdf_reader = PdfReader(file_path)
    text = "".join(page.extract_text() for page in pdf_reader.pages)
    return text

def test_chunking_methods(file_path):
    raw_text = preprocess_data(file_path)
    
    print("Testing Fixed-size Chunking...")
    fixed_chunks = chunk_data(raw_text, "Fixed-size Chunking", chunk_size=500)
    print(f"Number of Fixed-size Chunks: {len(fixed_chunks)}")
    
    print("Testing Semantic-based Chunking...")
    try:
        semantic_chunks = chunk_data(raw_text, "Semantic-based Chunking")
        print(f"Number of Semantic-based Chunks: {len(semantic_chunks)}")
    except Exception as e:
        print(f"Error in Semantic-based Chunking: {e}")
    
    print("Testing Single-linkage Clustering...")
    pre_chunks = chunk_data(raw_text, "Fixed-size Chunking", chunk_size=500)
    dummy_embeddings = [[0] * 128 for _ in pre_chunks]
    try:
        single_linkage_chunks = chunk_data(None, "Single-linkage Clustering", pre_chunks=pre_chunks, embeddings=dummy_embeddings)
        print(f"Number of Single-linkage Chunks: {len(single_linkage_chunks)}")
    except Exception as e:
        print(f"Error in Single-linkage Clustering: {e}")
    
    print("Testing DBSCAN Clustering...")
    try:
        dbscan_chunks = chunk_data(None, "DBSCAN Clustering", pre_chunks=pre_chunks, embeddings=dummy_embeddings)
        print(f"Number of DBSCAN Chunks: {len(dbscan_chunks)}")
    except Exception as e:
        print(f"Error in DBSCAN Clustering: {e}")
    
    print("Testing Breakpoint-based Chunking...")
    try:
        breakpoint_chunks = chunk_data(None, "Breakpoint-based Chunking", pre_chunks=pre_chunks, embeddings=dummy_embeddings)
        print(f"Number of Breakpoint-based Chunks: {len(breakpoint_chunks)}")
    except Exception as e:
        print(f"Error in Breakpoint-based Chunking: {e}")

if __name__ == "__main__":
    file_path = "/Users/poojithabalamurugan/Desktop/ece661_proj/data/reports/oxy-report.pdf"
    test_chunking_methods(file_path)
