import openai

def generate_final_response(aggregated_chunks, query, model="gpt-4o-mini"):
    """
    Generate the final response using aggregated chunks and the original query.

    Args:
    - aggregated_chunks: List of text chunks retrieved for sub-questions.
    - query: User's original query.
    - model: LLM to use for response generation.

    Returns:
    - Final response as a string.
    """
    context = "\n".join(aggregated_chunks)
    prompt = f"""
    You are an financial expert at analysing the SEC and 10k reports. Use the context from the SEC and 10k reports of the company below to answer the user's question concisely and accurately.
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
