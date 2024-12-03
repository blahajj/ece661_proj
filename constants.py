import os

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')  # OpenAI API key
CHROMADB_DIR = 'chromadb'  # Directory to store ChromaDB data
EMBEDDING_MODEL = 'text-embedding-3-small'  # Default embedding model
EMBEDDING_DIMENSIONS = None  # Set to desired dimensions or None for default