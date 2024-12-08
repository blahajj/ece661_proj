import openai

def generate_final_response(aggregated_chunks, query, document_type, model="gpt-4o-mini"):
    """
    Generate the final response using aggregated chunks and the original query.

    Args:
    - aggregated_chunks: List of text chunks retrieved for sub-questions.
    - query: User's original query.
    - model: LLM to use for response generation.

    Returns:
    - Final response as a string.
    """
    if document_type == "10K":
        document_type = "SEC 10K report"
    elif document_type == "Report":
        document_type == "Analyst Report"

    context = "\n".join(aggregated_chunks)
    prompt = f"""
    You are a seasoned equity analyst holding a CFA charter. 
    You are well capable of parsing financial foundamentals to answer high-level questions.
    You are aware that you shouldn't be giving direct financial advise (buy/sell).
    Use the context from the {document_type} of the company below to answer the user's question concisely and accurately.
    Make sure your answers also include evidence from the context to support your response.
    Context:
    {context}

    Question:
    {query}
    """
    response = openai.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                    {"role": "user", "content": prompt}
                    ],
                    temperature=0.7,
                )
    return response.choices[0].message.content.strip()
