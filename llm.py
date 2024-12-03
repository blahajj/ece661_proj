import openai

def get_llm_response(relevant_chunks):
    # Concatenate chunks for LLM
    context = " ".join(relevant_chunks)
    prompt = f"Answer the user's query based on the following context:\n\n{context}"
    
    response = openai.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                    {"role": "user", "content": prompt}
                    ],
                    temperature=0.7,
                )
    return response.choices[0].message.content
