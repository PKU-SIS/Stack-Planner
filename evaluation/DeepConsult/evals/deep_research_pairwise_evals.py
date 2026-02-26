import argparse
import concurrent.futures
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from tqdm import tqdm

from evals.metrics.deep_research_pairwise_metric import (
    DEFAULT_EVAL_MODEL,
    DeepResearchPairwiseMetric,
    DeepResearchScoreResult,
)


class DeepResearchEvaluator:
    """Evaluator class for evaluating deep reports using the DeepResearchPairwiseMetric."""

    def __init__(
        self,
        model: str = DEFAULT_EVAL_MODEL,
        output_path: Optional[Path] = None,
        num_workers: int = 4,
        metric_num_workers: int = 1,
        metric_num_trials: int = 3,
    ):
        """
        Initialize the evaluator.

        Args:
            model: The model to use for evaluation
            output_path: Path to save evaluation results
            num_workers: Number of workers for parallel processing of evaluation tasks
            metric_num_workers: Number of workers for the underlying pairwise metric
            metric_num_trials: Number of trials to run for each evaluation
        """
        self.model = model
        self.output_path = output_path
        self.num_workers = num_workers
        self.metric_num_trials = metric_num_trials

        # Initialize the pairwise evaluator
        self.pairwise_metric = DeepResearchPairwiseMetric(
            eval_model=model,
            num_trials=metric_num_trials,
            num_workers=metric_num_workers,  # Number of workers for the metric
        )

    def evaluate_single(
        self,
        question: str,
        baseline_answer: str,
        candidate_answer: str,
    ) -> Dict[str, Any]:
        """
        Evaluate a single question-answer pair.

        Args:
            question: The research question
            baseline_answer: The reference answer
            candidate_answer: The predicted answer to evaluate

        Returns:
            Dictionary containing evaluation results
        """
        result = {
            "question": question,
            "baseline_answer": baseline_answer,
            "candidate_answer": candidate_answer,
        }

        try:
            # Score the answer using the pairwise evaluator
            score_result = self.pairwise_metric.score(
                question=question,
                baseline_answer=baseline_answer,
                candidate_answer=candidate_answer,
            )

            # Add scores to the result
            result["success"] = True
            result["score_result"] = score_result.model_dump()

            # Add individual dimension scores for easier analysis
            for dimension in score_result.model_dump():
                dim_data = getattr(score_result, dimension)
                result[f"{dimension}_score"] = dim_data.score
                result[f"{dimension}_grade"] = dim_data.grade
                result[f"{dimension}_is_win"] = dim_data.is_win
                result[f"{dimension}_is_tie"] = dim_data.is_tie
                result[f"{dimension}_is_lose"] = dim_data.is_lose

        except Exception as e:
            result["success"] = False
            result["error"] = str(e)

        return result

    def evaluate_batch(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Evaluate a batch of question-answer pairs.

        Args:
            data: DataFrame containing questions and answers to evaluate
                Expected columns: 'question', 'baseline_answer', 'candidate_answer'

        Returns:
            List of evaluation results
        """
        results = []

        # Check if required columns exist
        required_columns = ["question", "baseline_answer", "candidate_answer"]
        for col in required_columns:
            if col not in data.columns:
                raise ValueError(f"Input data must contain column: {col}")

        # Create a list of tasks
        tasks = []
        for _, row in data.iterrows():
            tasks.append(
                {
                    "question": row["question"],
                    "baseline_answer": row["baseline_answer"],
                    "candidate_answer": row["candidate_answer"],
                }
            )

        # Process tasks in parallel
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self.num_workers
        ) as executor:
            # Submit all tasks
            futures = [
                executor.submit(
                    self.evaluate_single,
                    task["question"],
                    task["baseline_answer"],
                    task["candidate_answer"],
                )
                for task in tasks
            ]

            # Process results in order they were submitted
            for future in tqdm(
                futures,
                total=len(futures),
                desc="Evaluating",
            ):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    print(f"Error processing task: {e}")
                    results.append({"success": False, "error": str(e)})

        return results

    def aggregate_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Aggregate evaluation results.

        Args:
            results: List of evaluation results

        Returns:
            Dictionary containing aggregated metrics
        """
        # Filter successful evaluations
        successful_results = [r for r in results if r.get("success", False)]

        if not successful_results:
            return {"support": 0, "error": "No successful evaluations found"}

        # Convert results to DeepResearchScoreResult objects
        score_results = []
        for result in successful_results:
            try:
                score_result = DeepResearchScoreResult.model_validate(
                    result["score_result"]
                )
                score_results.append(score_result)
            except Exception as e:
                print(f"Error parsing score result: {e}")

        # Use the evaluator's aggregate method to get aggregate metrics
        return self.pairwise_metric.aggregate(score_results)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Deep Research pairwise evaluations"
    )
    parser.add_argument(
        "--input-data",
        type=str,
        default="datasets/DeepConsult/responses_OpenAI-DeepResearch_vs_ARI_2025-05-15.csv",
        help="Path to input CSV file. The CSV should have columns 'question', 'baseline_answer', and 'candidate_answer'.",
    )
    parser.add_argument(
        "--output-dir", type=str, required=True, help="Directory to save results"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_EVAL_MODEL,
        help="Model to use for evaluation",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of worker threads for evaluation",
    )
    parser.add_argument(
        "--metric-num-workers",
        type=int,
        default=1,
        help="Number of worker threads used in the pairwise metric computation on each row",
    )
    parser.add_argument(
        "--metric-num-trials",
        type=int,
        default=3,
        help="Number of trials per metric computation. Each trial runs the evaluation twice (with original and flipped inputs). Higher values produce more stable metrics but increase computation time.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Create output directory if it doesn't exist
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Set up output path
    output_path = output_dir / f"deep_research_results_{args.model}.jsonl"

    # Load input data
    print(f"Loading data from {args.input_data}")
    df = pd.read_csv(args.input_data)
    print(f"Loaded {len(df)} examples")

    # Initialize evaluator
    evaluator = DeepResearchEvaluator(
        model=args.model,
        output_path=output_path,
        num_workers=args.num_workers,
        metric_num_workers=args.metric_num_workers,
        metric_num_trials=args.metric_num_trials,
    )

    # Run evaluation
    print(
        f"Starting evaluation with model {args.model} using {args.num_workers} workers and {args.metric_num_workers} metric workers..."
    )
    results = evaluator.evaluate_batch(df)

    print(f"Results saved to {output_path}")
    pd.DataFrame(results).to_json(output_path, orient="records", lines=True)

    # Load and display summary
    results_df = pd.read_json(output_path, lines=True)
    print("\nResults summary:")
    print(f"Model: {args.model}")
    print(f"Total evaluations: {len(results_df)}")
    print(
        f"Successful evaluations: {len(results_df[results_df.get('success', False)])}"
    )
    print(f"Failed evaluations: {len(results_df[~results_df.get('success', False)])}")

    # Compute aggregate metrics
    if len(results) > 0:
        print("Aggregating results...")
        aggregate_metrics = evaluator.aggregate_results(results)

        # Save aggregate metrics
        aggregate_path = output_dir / f"deep_research_aggregate_{args.model}.json"
        with open(aggregate_path, "w") as f:
            json.dump(aggregate_metrics, f, indent=2)

        print(f"Aggregate metrics saved to {aggregate_path}")

        # Display key metrics
        print("\nKey Metrics:")
        print(f"Total examples: {aggregate_metrics.get('support', 0)}")

        if "overall" in aggregate_metrics:
            print("\nOverall Metrics:")
            for metric, value in aggregate_metrics["overall"].items():
                if isinstance(value, float):
                    print(f"{metric}: {value:.4f}")
                else:
                    print(f"{metric}: {value}")


if __name__ == "__main__":
    main()
