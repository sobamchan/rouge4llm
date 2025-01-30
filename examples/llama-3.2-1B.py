from rouge4llm.evaluator import EvaluationRunner, load_aclsum_evaluator
from rouge4llm.summarizer import LLaMASummarizer
from rouge4llm.utils import (
    AspectType,
    SplitType,
)

if __name__ == "__main__":
    evaluator = load_aclsum_evaluator(AspectType.challenge, SplitType.test)
    summarizer = LLaMASummarizer.load(
        "meta-llama/Llama-3.2-1B-Instruct",
        system_instruction="Generate a one-senetnce summary of the given research paper with a focus on its research challenge in less than 25 words.",
    )
    runner = EvaluationRunner(evaluator)

    result, _ = runner.run(summarizer)

    print(result)
