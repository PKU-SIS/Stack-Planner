
from ragas.dataset_schema import SingleTurnSample
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import ContextRelevance
from langchain_core.language_models import BaseLanguageModel

from typing import List

class RetrievalQualityEvaluator:
    def __init__(self, llm: BaseLanguageModel):
        evaluator_llm = LangchainLLMWrapper(llm)
        self.scorer = ContextRelevance(llm=evaluator_llm)

    def evaluate_retrieval(self, user_input: str, retrieved_contexts: List[str]):
        """Context Relevance evaluates whether the retrieved_contexts (chunks or passages) are pertinent to the user_input. This is done via two independent "LLM-as-a-judge" prompt calls that each rate the relevance on a scale of 0, 1, or 2. The ratings are then converted to a [0,1] scale and averaged to produce the final score. Higher scores indicate that the contexts are more closely aligned with the user's query.

            0 → The retrieved contexts are not relevant to the user’s query at all.

            1 → The contexts are partially relevant.
            
            2 → The contexts are completely relevant.

            Step 1: The LLM is prompted with two distinct templates (template_relevance1 and template_relevance2) to evaluate the relevance of the retrieved contexts concerning the user's query. Each prompt returns a relevance rating of 0, 1, or 2.

            Step 2: Each rating is normalized to a [0,1] scale by dividing by 2. If both ratings are valid, the final score is the average of these normalized values; if only one is valid, that score is used.
        """
        sample = SingleTurnSample(
            user_input=user_input,
            retrieved_contexts=retrieved_contexts
        )

        score = self.scorer.single_turn_score(sample)

        return score


if __name__ == "__main__":
    from langchain_openai import ChatOpenAI
    
    llm = ChatOpenAI(
        # model="qwen-turbo",
        model="qwen2.5-1.5b-instruct",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )

    user_input = "When and Where Albert Einstein was born?"

    rag_evaluator = RetrievalQualityEvaluator(llm)

    # completely relevant
    retrieved_contexts_1 = [
        "Albert Einstein was born March 14, 1879.",
        "Albert Einstein was born at Ulm, in Württemberg, Germany.",
    ]
    evaluation_result = rag_evaluator.evaluate_retrieval(user_input, retrieved_contexts_1)
    print(evaluation_result)

    # not relevant
    retrieved_contexts_2 = [
        "Wikis are powered by wiki software, also known as wiki engines.",
        "There are hundreds of thousands of wikis in use, both public and private.",
    ]
    evaluation_result = rag_evaluator.evaluate_retrieval(user_input, retrieved_contexts_2)
    print(evaluation_result)

    # partially relevant
    retrieved_contexts_3 = [
        "Born in the German Empire, Einstein moved to Switzerland in 1895",
        "Wikis are powered by wiki software, also known as wiki engines.",
    ]
    evaluation_result = rag_evaluator.evaluate_retrieval(user_input, retrieved_contexts_3)
    print(evaluation_result)