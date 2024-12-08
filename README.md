# Multi-Layer RAG for Financial Investment Screening

This repository contains the implementation of a **Retrieval-Augmented Generation (RAG)** pipeline for analyzing SEC and 10K financial documents. The project addresses key limitations of large language models (LLMs) in financial analysis, such as hallucination, limited context size, and difficulty in handling complex queries.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Evaluation Results](#evaluation-results)
- [Contributors](#contributors)

---

## Overview
The project focuses on leveraging a multi-layer RAG pipeline for strategic financial investment analysis. By integrating LLMs with advanced retrieval techniques, the pipeline ensures accurate and reliable answers to both broad and foundational financial questions.

### Key Goals
1. **Overcome LLM limitations**: Handle hallucinations, complex parsing, and restricted context size.
2. **Sub-question generation**: Enhance query precision with sub-questions.
3. **Modular workflow**: Allow customizable and reusable components for financial document analysis.

---

## Features
- **Financial Document Processing**: Chunking SEC and 10K documents into manageable pieces for retrieval.
- **Chunking Methods**: 
  - Fixed-size Chunking
  - Sentence-based Chunking
- **Retrieval with ChromaDB**: Indexing and retrieving relevant document chunks based on cosine similarity.
- **Sub-Question Generation**: Break down broad queries into specific sub-questions.
- **Evaluation Framework**: Metrics for response relevancy, context precision, and faithfulness using RAGAs.

---

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/blahajj/ece661_proj.git
   cd ece661_proj
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Set up environment variables:
   - Add your OpenAI API key in the `constants.py` file:
     ```python
     OPENAI_API_KEY = "your_api_key_here"
     ```

---

## Usage
### Running the Streamlit App
Start the Streamlit interface to process documents and generate responses:
```bash
streamlit run app.py
```

### Testing the Pipeline
Run the Jupyter notebook to evaluate performance:
```bash
jupyter notebook test.ipynb
```

---

## Project Structure
```
ECE661_PROJ/
├── chromadb/                   # ChromaDB database for indexing
├── data/                       # Directory for financial documents
├── env/                        # Environment configurations
├── testing/                    # Test scripts
├── app.py                      # Streamlit app
├── chunking.py                 # Chunking methods
├── constants.py                # API keys and constants
├── data_preprocessing.py       # Financial document preprocessing
├── evaluation_results_rag.csv  # Evaluation results
├── evaluation.py               # Evaluation metrics and scoring
├── question_generation.py      # Sub-question generation logic
├── response_generation.py      # LLM-based response generation
├── storing_retrieval.py        # Storing and retrieving chunks with ChromaDB
├── requirements.txt            # Python dependencies
├── README.md                   # Documentation
└── test.ipynb                  # Evaluation and testing notebook
```

---

## Evaluation Results
The evaluation results are saved in `evaluation_results_rag.csv`

For more details, see the `test.ipynb` notebook.

---

## Contributors
- **Poojitha Balamurugan**: Retrieval methods and pipeline implementation.
- **Henry Hai**: Architecture design and data processing.
- **Ritu Toshniwal**: Integration, deployment, and evaluation.

---

Contributions are welcomed.