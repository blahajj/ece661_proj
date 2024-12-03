import os
from datasets import load_dataset
from ragas import EvaluationDataset, evaluate
from ragas.metrics import LLMContextRecall, Faithfulness, FactualCorrectness, SemanticSimilarity
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from constants import OPENAI_API_KEY

# Set your OpenAI API Key
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# Initialize LLMs and embeddings
evaluator_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o-mini"))
evaluator_embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings())


def prepare_evaluation_dataset(llm_responses):
    """
    Prepares an evaluation dataset for RAGAS.

    Args:
    - llm_responses: List of dictionaries with "query", "llm_response", and "relevant_chunks".

    Returns:
    - EvaluationDataset object.
    """
    try:
        print("Preparing evaluation dataset...")

        # Ensure the context is a single string by joining chunks
        data = [
            {
                "question": response["query"],
                "context": " ".join(response["relevant_chunks"])
                if isinstance(response["relevant_chunks"], list)
                else response["relevant_chunks"],
                "answer": response["llm_response"]
            }
            for response in llm_responses
        ]

        # Debug data
        print("Data prepared for EvaluationDataset (sample):", data[:1])

        # Create EvaluationDataset
        eval_dataset = EvaluationDataset(data)
        print("EvaluationDataset created successfully")
        return eval_dataset
    except Exception as e:
        print("Error in prepare_evaluation_dataset:", e)
        raise



def evaluate_response(llm_responses):
    """
    Evaluates LLM responses using RAGAS metrics.

    Args:
    - llm_responses: List of dictionaries with "query", "llm_response", and "relevant_chunks".

    Returns:
    - DataFrame with evaluation metrics results.
    """
    try:
        print("Evaluating responses with RAGAS...")
        eval_dataset = prepare_evaluation_dataset(llm_responses)
        print("EvaluationDataset (sample):", eval_dataset[:1])

        # Define metrics
        metrics = [
            LLMContextRecall(llm=evaluator_llm),
            FactualCorrectness(llm=evaluator_llm),
            Faithfulness(llm=evaluator_llm),
            SemanticSimilarity(embeddings=evaluator_embeddings)
        ]
        print("Metrics initialized successfully")

        # Run evaluation
        results = evaluate(dataset=eval_dataset, metrics=metrics)
        print("Evaluation completed successfully")

        # Convert to DataFrame
        df = results.to_pandas()
        print("Results converted to DataFrame")
        return df
    except Exception as e:
        print("Error during evaluation:", e)
        raise

