import openai

def generate_sub_questions(query, model="gpt-4o-mini"):
    """
    Generate sub-questions from a broad query using an LLM.

    Args:
    - query: User's broad question.
    - model: LLM to use for question generation.

    Returns:
    - List of sub-questions.
    """
    prompt = f"""
    You are an financial expert assistant helping to refine a broad final question related to SEC document/ 10K document of the company. The user asked: "{query}".
    Generate 6 detailed sub-questions that will help retrieve more specific and useful chunks for my RAG from the vector database related to the user query.
    Make sure to only respond with questions that are relevant to the user query.
    Make sure that only questions are returned in the response in the following format:
    - Question 1
    - Question 2
    - Question 3
    - Question 4
    - Question 5
    - Question 6
    """
    response = openai.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                    {"role": "user", "content": prompt}
                    ],
                    temperature=0.7,
                )
    sub_questions = response.choices[0].message.content.strip().split("\n")
    return [q.strip("- ") for q in sub_questions if q.strip()]