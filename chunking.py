def chunk_data(raw_text, chunking_method):
    if chunking_method == "Fixed-size Chunking":
        # Chunk text into fixed sizes
        chunk_size = 500  # Example: 500 characters
        return [raw_text[i:i+chunk_size] for i in range(0, len(raw_text), chunk_size)]
    elif chunking_method == "Semantic-based Chunking":
        # Semantic chunking using NLP
        from transformers import pipeline
        summarizer = pipeline("summarization")
        return summarizer(raw_text, max_length=500, min_length=100, do_sample=False)
    else:
        raise ValueError("Unsupported chunking method!")
