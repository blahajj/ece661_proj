import os
import streamlit as st
from data_preprocessing import preprocess_data
from chunking import chunk_data
from storing_retrieval import index_chunks, retrieve_similar_chunks
from llm import get_llm_response
from evaluation import evaluate_response
from question_generation import generate_sub_questions
from response_generation import generate_final_response

# Main Streamlit Application
st.title("Fin-RAG for SEC and 10K Documents")

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
chunking_techniques = [
    "Fixed-size Chunking", 
    "Semantic-based Chunking", 
    "Single-linkage Clustering", 
    "DBSCAN Clustering", 
    "Breakpoint-based Chunking"
]
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

            st.write(f"Type of raw_text after preprocess_data: {type(raw_text)}")
            


            # Chunk the document
            st.write("Chunking the document...")
            if selected_chunking in ["Single-linkage Clustering", "DBSCAN Clustering", "Breakpoint-based Chunking"]:
                # For clustering-based methods, pre-process chunks and embeddings
                print("error")
                pre_chunks = chunk_data(raw_text, "Fixed-size Chunking", chunk_size=500)
                
                print(type(pre_chunks))
                embeddings = preprocess_data(pre_chunks)  # Replace with embedding logic if necessary
                chunks = chunk_data(
                    raw_text=None,
                    chunking_method=selected_chunking,
                    pre_chunks=pre_chunks,
                    embeddings=embeddings
                )
            else:
                # For fixed-size or semantic-based chunking
                chunks = chunk_data(raw_text, selected_chunking)
            
            st.write(f"Number of chunks created: {len(chunks)}")
            st.write(f"Type of chunks: {type(chunks)}")

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
user_query = st.text_input("Enter your question:")
if user_query:
    try:
        # Layer 1: Generate Sub-Questions
        st.write("Generating sub-questions...")
        sub_questions = generate_sub_questions(user_query)
        st.write("Sub-Questions Generated:")
        for i, sq in enumerate(sub_questions, 1):
            st.write(f"{i}. {sq}")

        # Layer 2: Retrieve Chunks for Each Sub-Question
        st.write("Retrieving chunks for sub-questions...")
        collection_name = f"{selected_company}_{document_type}".lower().replace(" ", "_")
        aggregated_chunks = []
        for question in sub_questions:
            chunks, _ = retrieve_similar_chunks(question, collection_name, top_k=5)
            aggregated_chunks.extend(chunks)
        st.write(f"Total Chunks Retrieved: {len(aggregated_chunks)}")

        # Layer 3: Generate Final Response
        st.write("Generating final response...")
        final_response = generate_final_response(aggregated_chunks, user_query)
        st.subheader("Final Response:")
        st.write(final_response)

    except Exception as e:
        st.error(f"Error retrieving or processing query: {e}")
