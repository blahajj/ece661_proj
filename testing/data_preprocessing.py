import os
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_community.vectorstores import Chroma

def load_documents(file_path):
    """Load documents from a PDF file."""
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    print(f"Loaded {len(documents)} documents.")
    return documents

def naive_chunking(documents, chunk_size=1000, chunk_overlap=0):
    """Split documents using naive chunking with RecursiveCharacterTextSplitter."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False
    )
    naive_chunks = text_splitter.split_documents(documents)
    print(f"Naive chunking resulted in {len(naive_chunks)} chunks.")
    return naive_chunks

def semantic_chunking(documents, embed_model, breakpoint_threshold_type="percentile"):
    """Split documents using semantic chunking."""
    semantic_chunker = SemanticChunker(embed_model, breakpoint_threshold_type=breakpoint_threshold_type)
    semantic_chunks = semantic_chunker.create_documents([d.page_content for d in documents])
    print(f"Semantic chunking resulted in {len(semantic_chunks)} chunks.")
    return semantic_chunks

def create_or_load_vectorstore(chunks, embed_model, persist_directory="../data/vector_database"):
    """Create or load a vectorstore from document chunks."""
    if os.path.exists(persist_directory):
        print("Loading existing vectorstore...")
        vectorstore = Chroma(persist_directory=persist_directory, embedding_function=embed_model)
    else:
        print("Creating new vectorstore...")
        vectorstore = Chroma.from_documents(chunks, embedding=embed_model, persist_directory=persist_directory)
        vectorstore.persist()  # Save the vectorstore to disk
    return vectorstore