import chromadb
from constants import CHROMADB_DIR, OPENAI_API_KEY, EMBEDDING_MODEL
import openai

# Initialize OpenAI API key
openai.api_key = OPENAI_API_KEY

def create_chroma_client():
    """
    Creates and returns a ChromaDB PersistentClient instance.
    """
    client = chromadb.PersistentClient(path=CHROMADB_DIR)
    return client

def get_embedding(text, model=EMBEDDING_MODEL):
    response = openai.embeddings.create(
        input=[text],
        model=model,
    )
    embedding = response.data[0].embedding
    return embedding

def index_chunks(chunks, collection_name):
    """
    Indexes chunks into a ChromaDB collection with embeddings.

    Args:
    - chunks: List of text chunks to be indexed.
    - collection_name: Name of the collection in ChromaDB.

    Returns:
    - The created collection.
    """
    client = create_chroma_client()
    collection = client.get_or_create_collection(collection_name)

    # Generate embeddings for chunks
    embeddings = [get_embedding(chunk) for chunk in chunks]
    ids = [str(i) for i in range(len(chunks))]

    # Add chunks to the collection
    collection.add(documents=chunks, embeddings=embeddings, ids=ids)
    return collection

def retrieve_similar_chunks(query, collection_name, top_k=5):
    """
    Retrieves the most similar chunks for a given query from a ChromaDB collection.

    Args:
    - query: The user query string.
    - collection_name: The name of the ChromaDB collection.
    - top_k: Number of top similar results to retrieve.

    Returns:
    - List of the most similar documents and their IDs.
    """
    client = create_chroma_client()
    collection = client.get_collection(collection_name)

    # Generate embedding for the query
    query_embedding = get_embedding(query)

    # Query the collection
    results = collection.query(query_embeddings=[query_embedding], n_results=top_k)
    return results['documents'][0], results['ids'][0],[1 - (distance / 2) for distance in results['distances'][0]] 
