import nltk

def chunk_data(raw_text, chunking_method):
    if chunking_method == "Fixed-size Chunking":
        # Chunk text into fixed sizes
        chunk_size = 5000
        return [raw_text[i:i+chunk_size] for i in range(0, len(raw_text), chunk_size)]
    elif chunking_method == "Sentence-based Chunking":
        # Chunk text into sentences
        import nltk
        nltk.download("punkt_tab", quiet=True)
        from nltk.tokenize import sent_tokenize
        
        sentences = sent_tokenize(raw_text)
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            if current_length + len(sentence) <= 5000:
                current_chunk.append(sentence)
                current_length += len(sentence)
            else:
                chunks.append(" ".join(current_chunk))
                current_chunk = [sentence]
                current_length = len(sentence)
        
        # Add the last chunk if any sentences remain
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        
        return chunks
    else:
        raise ValueError("Unsupported chunking method!")

