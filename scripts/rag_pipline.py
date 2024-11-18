import os
import sys
from data_preprocessing import (
    load_documents, 
    naive_chunking, 
    semantic_chunking, 
    create_vectorstore
)
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings

def define_rag_prompt():
    """Retrieval-Augmented Generation (RAG) prompt template."""
    rag_template = """\
    You are an expert Financial Analyst, and your goal is to assist in analyzing financial information and answering specific questions based solely on 
    the provided retrieval-augmented generation (RAG) context. You will not rely on external knowledge or make assumptions—your answers should strictly 
    adhere to the provided text. If the information required to answer a question is not present in the given context, clearly state that the data is unavailable. 
    Follow the structure and instructions below to ensure accuracy and relevance:
    User's Query:
    {question}

    Context:
    {context}

    Instructions:
    Strict Context Adherence:
    - Do not include information outside the supplied context.
    - Avoid generalizing or speculating if the context does not fully address a question.

    Structured Answers:
    - Provide concise, factual, and context-based responses.
    - Use bullet points or numbered lists where applicable for clarity.

    Indicate Missing Data:
    - If the question cannot be answered with the given context, explicitly state: "The provided context does not include the information necessary to answer this question."

    Maintain Professional Tone:
    - Your tone should be precise, analytical, and aligned with industry standards.

    Example Questions:
    Here are some examples of the questions you might be asked:

    What are the key revenue drivers for the company?
    How does the company's net profit margin compare to industry benchmarks?
    What are the identified risks in the current fiscal strategy?
    Can you summarize the financial health of the company based on the income statement?
    What are the major cost components impacting profitability?
    How has revenue growth trended over the past three fiscal years?

    Response Format:
    Restate the Question: Begin each response by paraphrasing the question to confirm understanding.
    Provide Evidence-Based Analysis: Use relevant data points or text excerpts from the RAG context to support your analysis.
    Highlight Assumptions or Gaps: Note any missing details explicitly if the context does not allow a full answer.

    Example Interaction:
    Question: What were the primary factors contributing to the company's revenue growth last year?
    Context Provided:
    Revenue increased by 15% due to higher product sales and expansion into new markets.
    A 5% price increase was implemented mid-year.
    Marketing expenses doubled, resulting in greater brand awareness and customer acquisition.
    Answer:
    The primary factors contributing to the company's revenue growth last year include:
    - A 15% overall increase in revenue, driven primarily by higher product sales and geographical market expansion.
    - A 5% price increase implemented mid-year, which likely improved per-unit revenue.
    - Doubling of marketing expenses, leading to enhanced brand visibility and higher customer acquisition rates.
    """
    rag_prompt = ChatPromptTemplate.from_template(rag_template)
    return rag_prompt

def create_chat_model():
    """Initialize the chat model."""
    groq_api_key = os.environ.get('GROQ_API_KEY')
    if not groq_api_key:
        raise ValueError("Please set the 'GROQ_API_KEY' environment variable.")
    chat_model = ChatGroq(
        temperature=0,
        model_name="mixtral-8x7b-32768",
        api_key=groq_api_key,
    )
    return chat_model

def create_retriever(vectorstore, k=5):
    """Create a retriever from the vectorstore."""
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    return retriever

def create_rag_chain(prompt, chat_model):
    """Create the RAG chain without retriever (context is supplied directly)."""
    rag_chain = (
        {"context": RunnablePassthrough(), "question": RunnablePassthrough()}
        | prompt
        | chat_model
        | StrOutputParser()
    )
    return rag_chain


if __name__ == "__main__":
    # Set the path to your PDF file
    file_path = "../data/reports/axp-report.pdf"

    # Load documents
    documents = load_documents(file_path)

    # Initialize embedding model
    embed_model = FastEmbedEmbeddings(model_name="BAAI/bge-base-en-v1.5")

    # Choose chunking method ('naive' or 'semantic')
    chunking_method = sys.argv[1] if len(sys.argv) > 1 else 'naive'

    if chunking_method == 'naive':
        chunks = naive_chunking(documents)
    elif chunking_method == 'semantic':
        chunks = semantic_chunking(documents, embed_model)
    else:
        print("Invalid chunking method. Choose 'naive' or 'semantic'.")
        sys.exit(1)

    # Create vectorstore and retriever
    vectorstore = create_vectorstore(chunks, embed_model)
    retriever = create_retriever(vectorstore, k=5)

    # Define prompt and model
    rag_prompt = define_rag_prompt()
    chat_model = create_chat_model()

    # Create RAG chain
    rag_chain = create_rag_chain(rag_prompt, chat_model)

    # List of queries to run
    queries = [
        "What was earnings per share in the third quarter of 2023?",
        "What is the updated fair value estimate for American Express, and how does it compare to the previous estimate in terms of projected 2024 earnings?",
        "What is the Opportunity in Credit Card Issuers?",
        "What is the projected net interest income CAGR for the company from 2023 to 2028?"
    ]

    # Run queries
    for query in queries:
        # Retrieve relevant documents
        retrieved_docs = retriever.get_relevant_documents(query)
        
        # Print the retrieved documents
        print(f"Retrieved documents for query '{query}':\n")
        for idx, doc in enumerate(retrieved_docs):
            print(f"Document {idx+1}:\n{doc.page_content}\n{'-'*20}\n")
        
        # Prepare the context by combining the retrieved documents
        context = "\n".join([doc.page_content for doc in retrieved_docs])
        
        # Now invoke the chain with context and question
        response = rag_chain.invoke({"context": context, "question": query})
        print(f"Question: {query}\n")
        print(f"Answer: {response}\n{'-'*50}\n")