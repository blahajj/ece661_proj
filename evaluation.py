import os
from datasets import Dataset
from ragas.metrics import (
    ResponseRelevancy,
    LLMContextPrecisionWithoutReference,
    Faithfulness,
)
from ragas import evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from constants import OPENAI_API_KEY

# Set OpenAI API key
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# Initialize LLMs and Embeddings
evaluator_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o-mini"))
evaluator_embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings())


def prepare_evaluation_dataset(questions, answers, contexts, ground_truths=None):
    """
    Prepare the dataset for RAGAs evaluation.

    """
    # Ensure contexts is a list of lists
    if not all(isinstance(ctx, list) for ctx in contexts):
        contexts = [ctx if isinstance(ctx, list) else [ctx] for ctx in contexts]

    data = {
        "user_input": questions,
        "response": answers,
        "retrieved_contexts": contexts, 
    }
    if ground_truths:
        data["reference"] = ground_truths
    return Dataset.from_dict(data)


def evaluate_response_with_ragas(questions, answers, contexts, ground_truths=None):
    """
    Evaluate the RAG pipeline using RAGAs metrics.
    """
    dataset = prepare_evaluation_dataset(questions, answers, contexts, ground_truths)

    # Define metrics for evaluation
    metrics = [
        ResponseRelevancy(llm=evaluator_llm),  # Relevancy of the response to the question
        Faithfulness(llm=evaluator_llm),  # Faithfulness of the response to the retrieved context
        LLMContextPrecisionWithoutReference(llm=evaluator_llm),  # Precision of the context in the response
    ]

    # Perform evaluation
    scores = evaluate(dataset=dataset, metrics=metrics)
    scores.to_pandas()
    return scores



