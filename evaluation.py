from langchain_google_vertexai import ChatVertexAI
from ragas import evaluate
from datasets import Dataset
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)

from app.services.rag_service import ask_question


def run_evaluation():

    # ✅ Example test dataset
    questions = [
        "What is machine learning?",
        "Explain deep learning",
    ]

    ground_truths = [
        "Machine learning is a branch of AI...",
        "Deep learning is a subset of machine learning...",
    ]

    answers = []
    contexts = []

    for q in questions:
        answer, sources = ask_question(q, session_id="eval")
        answers.append(answer)

        # ✅ get context manually if needed
        contexts.append(["Context not captured yet"])

    dataset = Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths
    })

    result = evaluate(
        dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        ],
    )

    print(result)