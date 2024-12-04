import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_community.vectorstores import Chroma
from sklearn.cluster import AgglomerativeClustering, DBSCAN
import numpy as np
from langchain.embeddings.base import Embeddings
from transformers import AutoTokenizer, AutoModel
import torch

from langchain.embeddings.base import Embeddings
from transformers import AutoTokenizer, AutoModel
import torch

class CustomEmbeddingModel(Embeddings):
    """Custom embedding model using Hugging Face Transformers."""
    def __init__(self, model_name: str = "dunzhang/stella_en_1.5B_v5"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

    def _embed(self, texts):
        """Helper method to generate embeddings for a list of texts."""
        if isinstance(texts, str):
            texts = [texts]
        inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1).tolist()
        return embeddings

    def embed_documents(self, texts):
        """Embed a list of documents."""
        return self._embed(texts)

    def embed_query(self, text):
        """Embed a single query."""
        return self._embed(text)


def load_documents(file_path):
    """Load documents from a PDF file."""
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    print(f"Loaded {len(documents)} documents.")
    return documents

def embed_sentences(sentences, embed_model):
    """Generate embeddings for a list of sentences."""
    raw_embeddings = embed_model.embed_documents(sentences)  # Get raw embeddings
    # Ensure embeddings are processed correctly
    embeddings = [
        np.mean(np.array(embed), axis=0) if isinstance(embed, list) else embed
        for embed in raw_embeddings
    ]
    # Convert to 2D array
    embeddings = np.array(embeddings)
    if embeddings.ndim == 1:  # If embeddings are 1D, reshape to 2D
        embeddings = embeddings.reshape(1, -1)
    return embeddings


def single_linkage_clustering(embeddings, max_chunk_size=10):
    """Perform Single-linkage Agglomerative Clustering."""
    clustering = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=0.5,  # Threshold to decide clusters
        linkage="single"
    )
    labels = clustering.fit_predict(embeddings)
    return labels

def dbscan_clustering(embeddings, eps=0.5, min_samples=2):
    """Perform DBSCAN Clustering."""
    clustering = DBSCAN(eps=eps, min_samples=min_samples, metric="cosine")
    labels = clustering.fit_predict(embeddings)
    return labels

def breakpoint_based_chunking(sentences, embeddings, threshold=0.5):
    """Perform Breakpoint-based Chunking."""
    chunks = []
    current_chunk = [sentences[0]]
    for i in range(1, len(sentences)):
        similarity = 1 - np.dot(embeddings[i - 1], embeddings[i])  # Cosine similarity
        if similarity > threshold:
            chunks.append(" ".join(current_chunk))
            current_chunk = []
        current_chunk.append(sentences[i])
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return chunks

def create_vectorstore(chunks, embed_model, persist_directory):
    """Create a vectorstore from document chunks."""
    vectorstore = Chroma.from_texts(chunks, embedding=embed_model, persist_directory=persist_directory)
    return vectorstore

if __name__ == "__main__":
    file_path = "../data/reports/sample_report.pdf"

    # Load documents
    documents = load_documents(file_path)
    sentences = [d.page_content for d in documents]

    # Initialize custom embedding model
    embed_model = CustomEmbeddingModel(model_name="dunzhang/stella_en_1.5B_v5")

    # Generate 2D embeddings
    sentence_embeddings = embed_sentences(sentences, embed_model)

    # Handle single input
    if sentence_embeddings.shape[0] < 2:
        print("Only one sentence detected; clustering cannot proceed.")
        single_linkage_labels = [0]  # Assign a default label
    else:
        # Single-linkage Agglomerative Clustering
        single_linkage_labels = single_linkage_clustering(sentence_embeddings)

    single_linkage_chunks = [
        " ".join([sentences[i] for i in range(len(sentences)) if single_linkage_labels[i] == label])
        for label in set(single_linkage_labels)
    ]

    print(f"Single-linkage Clustering Chunks: {single_linkage_chunks}")

    print(f"Single-linkage Clustering Chunks: {single_linkage_chunks}")

    # DBSCAN Clustering
    # Perform DBSCAN Clustering
    dbscan_labels = dbscan_clustering(sentence_embeddings, eps=0.5, min_samples=2)

    # Check if there are valid clusters
    if len(dbscan_labels) == 0 or all(label == -1 for label in dbscan_labels):
        print("No valid clusters formed by DBSCAN.")
        dbscan_chunks = []  # No valid clusters
    else:
        dbscan_chunks = [
            " ".join([sentences[i] for i in range(len(sentences)) if dbscan_labels[i] == label])
            for label in set(dbscan_labels) if label != -1
        ]

    # Create vectorstore only if dbscan_chunks is not empty
    if dbscan_chunks:
        dbscan_vectorstore = create_vectorstore(dbscan_chunks, embed_model, "../data/vector_db/dbscan")
        print("DBSCAN vector store created successfully.")
    else:
        print("No valid DBSCAN chunks to create vector store.")

    # Breakpoint-based Chunking
    breakpoint_chunks = breakpoint_based_chunking(sentences, sentence_embeddings, threshold=0.5)

    # Store chunks in vector databases
    single_linkage_vectorstore = create_vectorstore(single_linkage_chunks, embed_model, "../data/vector_db/single_linkage")
    
    breakpoint_vectorstore = create_vectorstore(breakpoint_chunks, embed_model, "../data/vector_db/breakpoint")

    print("Vector stores created for Single-linkage, DBSCAN, and Breakpoint-based chunking.")
