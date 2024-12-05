import numpy as np
from sklearn.cluster import AgglomerativeClustering, DBSCAN
from transformers import pipeline

def chunk_data(raw_text, chunking_method, **kwargs):
    """
    Chunk raw text into different chunking methods.

    Parameters:
        raw_text (str): The text to be chunked.
        chunking_method (str): The chunking method to use.
        kwargs: Additional parameters for chunking methods.

    Returns:
        list: List of chunked texts.
    """
    if chunking_method == "Fixed-size Chunking":
        # Chunk text into fixed sizes
        chunk_size = kwargs.get("chunk_size", 500)
        return [raw_text[i:i+chunk_size] for i in range(0, len(raw_text), chunk_size)]
    
    elif chunking_method == "Semantic-based Chunking":
        # Semantic chunking using NLP
        summarizer = pipeline("summarization")
        return summarizer(raw_text, max_length=500, min_length=100, do_sample=False)
    
    elif chunking_method == "Single-linkage Clustering":
        # Perform single-linkage clustering
        pre_chunks = kwargs.get("pre_chunks")
        embeddings = kwargs.get("embeddings")
        if not pre_chunks or not embeddings:
            raise ValueError("pre_chunks and embeddings are required for Single-linkage Clustering.")
        
        clustering = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=0.5,
            linkage="single"
        )
        labels = clustering.fit_predict(embeddings)
        return [
            " ".join([pre_chunks[i] for i in range(len(pre_chunks)) if labels[i] == label])
            for label in set(labels)
        ]

    elif chunking_method == "DBSCAN Clustering":
        # Perform DBSCAN clustering
        pre_chunks = kwargs.get("pre_chunks")
        embeddings = kwargs.get("embeddings")
        eps = kwargs.get("eps", 0.3)
        min_samples = kwargs.get("min_samples", 1)
        if not pre_chunks or not embeddings:
            raise ValueError("pre_chunks and embeddings are required for DBSCAN Clustering.")
        
        clustering = DBSCAN(eps=eps, min_samples=min_samples, metric="cosine")
        labels = clustering.fit_predict(embeddings)
        return [
            " ".join([pre_chunks[i] for i in range(len(pre_chunks)) if labels[i] == label])
            for label in set(labels) if label != -1
        ]

    elif chunking_method == "Breakpoint-based Chunking":
        # Perform breakpoint-based chunking
        pre_chunks = kwargs.get("pre_chunks")
        embeddings = kwargs.get("embeddings")
        threshold = kwargs.get("threshold", 0.5)
        if not pre_chunks or not embeddings:
            raise ValueError("pre_chunks and embeddings are required for Breakpoint-based Chunking.")
        
        chunks = []
        current_chunk = [pre_chunks[0]]
        for i in range(1, len(pre_chunks)):
            similarity = 1 - np.dot(embeddings[i - 1], embeddings[i])  # Cosine similarity
            if similarity > threshold:
                chunks.append(" ".join(current_chunk))
                current_chunk = []
            current_chunk.append(pre_chunks[i])
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        return chunks
    
    else:
        raise ValueError("Unsupported chunking method!")
