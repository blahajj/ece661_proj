import os
import streamlit as st
from data_preprocessing import preprocess_data
from chunking import chunk_data
from storing_retrieval import index_chunks, retrieve_similar_chunks
from llm import get_llm_response
from evaluation import evaluate_response

# Main Streamlit Application
st.title("RAG for SEC and 10K Documents Using ChromaDB")

# Define available company names and document types
company_names = [
    "AXP", "BAC", "CB", "CVX", "Itochu", 
    "KHC", "KO", "MCO", "Mitsubishi", "OXY"
]
document_types = ["10K", "Report"]

# User selections
selected_company = st.selectbox("Select a company:", company_names)
document_type = st.radio("Select document type:", document_types)

# Select chunking technique
chunking_techniques = ["Fixed-size Chunking", "Semantic-based Chunking"]
selected_chunking = st.radio("Select chunking technique:", chunking_techniques)

# Directory and file selection based on naming convention
data_dir = "data"
if document_type == "10K":
    folder = "10Ks"
    file_suffix = "-2024-10k.pdf"
else:
    folder = "reports"
    file_suffix = "-report.pdf"

file_path = os.path.join(data_dir, folder, f"{selected_company.lower()}{file_suffix}")

if st.button("Process Document"):
    st.write(f"Processing document: {file_path}...")
    
    try:
        with open(file_path, "rb") as file:
            # Preprocess the document
            raw_text = preprocess_data(file)

            # Chunk the document
            st.write("Chunking the document...")
            chunks = chunk_data(raw_text, selected_chunking)
            st.write(f"Number of chunks created: {len(chunks)}")

            # Store chunks in ChromaDB
            collection_name = f"{selected_company}_{document_type}".lower().replace(" ", "_")
            st.write("Indexing chunks into ChromaDB...")
            index_chunks(chunks, collection_name)
            st.success("Document processed and indexed successfully.")
    except FileNotFoundError:
        st.error(f"File not found: {file_path}")
    except Exception as e:
        st.error(f"Error processing document: {e}")

# Query Input
user_query = st.text_input("Enter your query:")
if user_query:
    st.write("Retrieving relevant chunks...")
    
    try:
        # Retrieve relevant chunks from ChromaDB
        collection_name = f"{selected_company}_{document_type}".lower().replace(" ", "_")
        relevant_chunks, chunk_ids = retrieve_similar_chunks(user_query, collection_name)
        relevant_chunks = [chunk if isinstance(chunk, str) else " ".join(chunk) for chunk in relevant_chunks]
        # Display relevant chunks
        st.write("Relevant Chunks:")
        for i, chunk in enumerate(relevant_chunks):
            st.write(f"Chunk {chunk_ids[i]}: {chunk}")

        # Get LLM response
        st.write("Generating response using LLM...")
        llm_response = get_llm_response(relevant_chunks)
        st.subheader("LLM Response:")
        st.write(llm_response)

        llm_responses = [{
        "query": user_query,
        "llm_response": llm_response,
        "relevant_chunks": " ".join(relevant_chunks)  # Join chunks into a single string
        }]

        # Evaluate the response
        st.write("Evaluating response using RAGAS...")
        evaluation_results = evaluate_response(llm_responses)
        st.subheader("Evaluation Results:")
        st.write(evaluation_results)

    except Exception as e:
        st.error(f"Error retrieving or processing query: {e}")
