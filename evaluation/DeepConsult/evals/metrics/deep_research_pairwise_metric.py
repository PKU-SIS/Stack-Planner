import concurrent.futures
import json
import random
from typing import Any, Dict, List, Literal

from openai.types.chat import ChatCompletionMessageParam
from pydantic import BaseModel, Field, computed_field, field_validator
from retry import retry

from evals.utils import (
    query_openai_model,
    query_openai_model_structured_outputs,
    replace_markdown_links_with_text,
)

DIMENSIONS = [
    "instruction_following",
    "comprehensiveness",
    "completeness",
    "writing_quality",
]

DEFAULT_EVAL_MODEL = "o3-mini-2025-01-31"


class Preference(BaseModel):
    """Represents a preference between two answers."""

    explanation: str
    preferred: Literal["a", "b"]
    gap_score: int

    @computed_field
    def score_b(self) -> int:
        return -self.gap_score if self.preferred == "a" else self.gap_score


class DeepResearchPairwisePreferenceOutput(BaseModel):
    """Represents the output of deep research pairwise preference evaluation."""

    instruction_following: Preference = Field(
        description="Evaluates response's fidelity to user specified instructions and constraints."
    )
    comprehensiveness: Preference = Field(
        description="Measures breadth and range of information covered in response, addressing the scope of user request."
    )
    completeness: Preference = Field(
        description="Measures the depth and thoroughness of information for topics addressed in the report."
    )
    writing_quality: Preference = Field(
        description="Evaluates clarity, conciseness, logical organization and overall readability of the report."
    )


class DeepResearchPairwisePreferenceInput(BaseModel):
    question: str
    baseline_answer: str
    candidate_answer: str

    @field_validator("candidate_answer", "baseline_answer")
    @classmethod
    def validate_non_empty_answer(cls, v: str) -> str:
        if not v or v.strip() == "":
            raise ValueError("Answer cannot be empty")
        v = replace_markdown_links_with_text(v, "")
        return v


DEEP_RESEARCH_PAIRWISE_PROMPT = """
You are an expert evaluator for reports to a research question. You'll be comparing two responses to a research question: report_a and report_b.

Evaluate both reports on these dimensions:
1. Instruction following: Evaluates response's fidelity to user specified instructions and constraints.
2. Comprehensiveness: Measures breadth and range of information covered in response, addressing the scope of user request.
3. Completeness: Measures the depth and thoroughness of information for topics addressed in the report.
4. Writing quality: Evaluates clarity, conciseness, logical organization and overall readability of the report.

For each dimension, indicate which report you prefer (either "a" or "b") and provide a concise explanation for your choice. 
Your explanations should cite specific examples to justify your preference and point out what can be improved in the other report.
Also provide a gap score that measures the difference in quality between the two reports for that dimension. 
The gap score should be a number from 0 to 5, where 0 indicates that both reports have similar quality and 5 is the maximum difference in quality.

Be fair and objective in your evaluation. Do not be biased towards either report A or B.
The length of a report is not necessarily an indicator of quality - focus on the substance and how well it meets the user's needs.
"""


class DimensionResult(BaseModel):
    """Represents the evaluation results for a single dimension."""

    grade: str
    is_win: bool
    is_tie: bool
    is_lose: bool
    score: float
    preferred: List[str]
    raw_preferences: dict


class DeepResearchScoreResult(BaseModel):
    """Represents the complete evaluation results across all dimensions."""

    instruction_following: DimensionResult
    comprehensiveness: DimensionResult
    completeness: DimensionResult
    writing_quality: DimensionResult


class DeepResearchPairwiseMetric:
    """Metric for deep research using pairwise comparison evaluations on research-style answers."""

    def __init__(
        self,
        eval_model: str = DEFAULT_EVAL_MODEL,
        num_trials: int = 3,
        num_workers: int = 3,
    ):
        """
        Initialize the evaluator.

        Args:
            eval_model: The model to use for evaluation
            num_trials: Number of times to run the evaluation model
            num_workers: Number of parallel workers to use for processing trials
        """
        self.eval_model = eval_model
        self.num_trials = num_trials
        self.num_workers = num_workers

    def _get_evaluation_messages(
        self, metric_input: DeepResearchPairwisePreferenceInput
    ) -> List[ChatCompletionMessageParam]:
        """Generate the messages for the evaluation model."""
        return [
            {"role": "system", "content": DEEP_RESEARCH_PAIRWISE_PROMPT},
            {
                "role": "user",
                "content": f"""
<prompt>
{metric_input.question}
</prompt>

<report_a>
{metric_input.baseline_answer}
</report_a>

<report_b>
{metric_input.candidate_answer}
</report_b>
""",
            },
        ]

    @retry(tries=3, delay=1, backoff=2)
    def _query_evaluation_model(
        self, messages: List[ChatCompletionMessageParam]
    ) -> DeepResearchPairwisePreferenceOutput:
        """Query the evaluation model with retry logic."""
        output = query_openai_model_structured_outputs(
            messages=messages,
            output_class=DeepResearchPairwisePreferenceOutput,
            model=self.eval_model,
            temperature=0,
            max_completion_tokens=10000,
        )
        if output is None:
            raise ValueError("Failed to get output from evaluation model")
        return DeepResearchPairwisePreferenceOutput.model_validate(output)

    def _get_pairwise_preference(
        self, metric_input: DeepResearchPairwisePreferenceInput
    ) -> dict:
        """Get pairwise preference between a baseline and candidate research answer."""
        # Create flipped input (baseline=B, candidate=A)
        input_flipped = DeepResearchPairwisePreferenceInput(
            question=metric_input.question,
            baseline_answer=metric_input.candidate_answer,
            candidate_answer=metric_input.baseline_answer,
        )

        # Get messages for both original and flipped inputs
        messages = self._get_evaluation_messages(metric_input)
        messages_flipped = self._get_evaluation_messages(input_flipped)

        # Run all trials for both original and flipped inputs
        all_outputs = []
        all_outputs_flipped = []

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self.num_workers
        ) as executor:
            # Submit all trial tasks for original input
            original_futures = [
                executor.submit(self._query_evaluation_model, messages)
                for _ in range(self.num_trials)
            ]

            # Submit all trial tasks for flipped input
            flipped_futures = [
                executor.submit(self._query_evaluation_model, messages_flipped)
                for _ in range(self.num_trials)
            ]

            # Collect results for original inputs
            for future in concurrent.futures.as_completed(original_futures):
                try:
                    output = future.result()
                    if output is not None:
                        all_outputs.append(output)
                except Exception as exc:
                    print(f"Original trial generated an exception: {exc}")

            # Collect results for flipped inputs
            for future in concurrent.futures.as_completed(flipped_futures):
                try:
                    output = future.result()
                    if output is not None:
                        all_outputs_flipped.append(output)
                except Exception as exc:
                    print(f"Flipped trial generated an exception: {exc}")

        if not all_outputs or not all_outputs_flipped:
            raise ValueError("Failed to get enough outputs from evaluation model")

        # Prepare results for each dimension
        results = {}
        for dimension in DIMENSIONS:
            all_preferred = []

            # Collect preferences from original and flipped evaluations
            preferences = [getattr(output, dimension) for output in all_outputs]
            preferences_flipped = [
                getattr(output, dimension) for output in all_outputs_flipped
            ]

            # Store preferences from original evaluations
            for pref in preferences:
                all_preferred.append(pref.preferred)

            # Store preferences from flipped evaluations
            for pref in preferences_flipped:
                # For flipped evaluations, we need to flip the preference too
                flipped_pref = "a" if pref.preferred == "b" else "b"
                all_preferred.append(flipped_pref)

            # Calculate consensus for this dimension
            num_wins = sum(1 for p in all_preferred if p == "b")
            num_losses = sum(1 for p in all_preferred if p == "a")

            # Determine grade based on majority vote across all trials
            if num_wins > num_losses:
                grade = "win"
            elif num_wins < num_losses:
                grade = "lose"
            else:
                grade = "tie"

            # Calculate consensus score (normalized to 0-10 scale)
            original_scores = [p.score_b for p in preferences]
            flipped_scores = [-p.score_b for p in preferences_flipped]
            all_scores = original_scores + flipped_scores
            consensus_score = sum(all_scores) / len(all_scores) + 5
            results[dimension] = {
                "raw_preferences": {
                    "original": [p.model_dump() for p in preferences],
                    "flipped": [p.model_dump() for p in preferences_flipped],
                },
                "all_preferred": all_preferred,
                "all_scores": all_scores,
                "consensus_grade": grade,
                "consensus_score": consensus_score,
            }

        return results

    def _generate_explanation_summary_from_raw(
        self, raw_preferences_list: List[Dict[str, Any]], n: int = 20
    ) -> str:
        """Generate a summary from raw preferences."""
        sampled_preferences = random.sample(
            raw_preferences_list, min(n, len(raw_preferences_list))
        )

        raw_preferences_str = "\n\n".join(
            [
                f"DIMENSION: {item['dimension']}\n{item['raw_preferences']['original']}"
                for item in sampled_preferences
            ]
        )

        # Query explanation summary using OpenAI model
        summary_messages: List[ChatCompletionMessageParam] = [
            {
                "role": "system",
                "content": "You are a helpful assistant that analyzes evaluation feedback.",
            },
            {
                "role": "user",
                "content": f"""
Below are evaluation results comparing two research reports (report_a and report_b).
Please analyze the feedback and provide a clear summary of both strengths and weaknesses of report_b compared to report_a.
Organize your analysis by dimension with separate bullet points for strengths and weaknesses.
Focus on specific, actionable insights that highlight what report_b does well and what could be improved.

<Example format>
# DIMENSION: instruction_following
STRENGTHS:
- report_b directly addresses the main question
- report_b follows the specified format requirements

WEAKNESSES:
- report_b misses some specific instructions mentioned in the query
- report_b includes unnecessary information not requested by the user

# DIMENSION: comprehensiveness
STRENGTHS:
- report_b covers several important aspects of the topic
- report_b includes relevant examples

WEAKNESSES:
- report_b omits key subtopics that report_a addresses
- report_b lacks sufficient breadth in its analysis

# DIMENSION: completeness
STRENGTHS:
- report_b provides detailed analysis on certain points
- report_b includes supporting evidence for its claims

WEAKNESSES:
- report_b lacks depth in critical areas
- report_b misses important details on several topics

# DIMENSION: writing_quality
STRENGTHS:
- report_b has clear paragraph structure
- report_b uses appropriate terminology

WEAKNESSES:
- report_b contains organizational issues that affect readability
- report_b would benefit from more concise phrasing
</Example format>

{raw_preferences_str}
""",
            },
        ]

        try:
            explanation_summary = query_openai_model(
                messages=summary_messages,
                model=self.eval_model,
                temperature=0,
                max_output_tokens=10000,
                timeout=240,
            )["content"]

            return explanation_summary or "Failed to generate explanation summary"
        except Exception as e:
            print(f"Failed to generate explanation summary: {e}")
            return f"Error generating summary: {str(e)}"

    def score(
        self,
        question: str,
        baseline_answer: str,
        candidate_answer: str,
    ) -> DeepResearchScoreResult:
        """
        Score a single question-answer pair.

        Args:
            question: The research question
            baseline_answer: The baseline answer (report_a)
            candidate_answer: The candidate answer (report_b)

        Returns:
            Object containing dimension-level scores and grades
        """
        metric_input = DeepResearchPairwisePreferenceInput(
            question=question,
            baseline_answer=baseline_answer,
            candidate_answer=candidate_answer,
        )

        output_dict = self._get_pairwise_preference(metric_input)

        # Create dimension results
        dimension_results = {}
        for dimension in DIMENSIONS:
            # Extract results for this dimension
            dimension_output_dict = output_dict[dimension]

            # Create DimensionResult for this dimension
            dimension_results[dimension] = DimensionResult(
                grade=dimension_output_dict["consensus_grade"],
                is_win=dimension_output_dict["consensus_grade"] == "win",
                is_tie=dimension_output_dict["consensus_grade"] == "tie",
                is_lose=dimension_output_dict["consensus_grade"] == "lose",
                score=dimension_output_dict["consensus_score"],
                preferred=dimension_output_dict["all_preferred"],
                raw_preferences=dimension_output_dict["raw_preferences"],
            )

        # Create and return the full result object
        return DeepResearchScoreResult(**dimension_results)

    def aggregate(self, scores_list: List[DeepResearchScoreResult]) -> Dict[str, Any]:
        """
        Aggregate metrics from multiple scored rows.

        Args:
            scores_list: List of score result objects from multiple rows

        Returns:
            Dictionary containing aggregated metrics values
        """
        # Initialize the aggregated metrics dictionary
        aggregated_metrics: Dict[str, Any] = {
            "support": len(scores_list),
        }

        # Calculate metrics for each dimension
        for dimension in DIMENSIONS:
            # Extract results for this dimension across all scores
            dimension_results = [
                getattr(score_result, dimension) for score_result in scores_list
            ]

            # Calculate metrics
            win_rate = sum(result.is_win for result in dimension_results) / len(
                dimension_results
            )
            tie_rate = sum(result.is_tie for result in dimension_results) / len(
                dimension_results
            )
            lose_rate = sum(result.is_lose for result in dimension_results) / len(
                dimension_results
            )
            avg_score = sum(result.score for result in dimension_results) / len(
                dimension_results
            )

            # Calculate net winrate
            num_wins = sum(result.is_win for result in dimension_results)
            num_losses = sum(result.is_lose for result in dimension_results)
            net_winrate = (
                num_wins / (num_wins + num_losses)
                if (num_wins + num_losses) > 0
                else 0.0
            )

            # Store in dimension-specific dict
            aggregated_metrics[dimension] = {
                "win_rate": win_rate,
                "tie_rate": tie_rate,
                "lose_rate": lose_rate,
                "avg_score": avg_score,
                "net_winrate": net_winrate,
            }

        # Create overall average dictionaries
        aggregated_metrics["overall"] = {}

        # Calculate overall averages for each metric
        metrics = [
            "win_rate",
            "tie_rate",
            "lose_rate",
            "avg_score",
            "net_winrate",
        ]
        for metric in metrics:
            overall_avg = sum(
                aggregated_metrics[dimension][metric] for dimension in DIMENSIONS
            ) / len(DIMENSIONS)
            aggregated_metrics["overall"][metric] = overall_avg

        # Collect raw preferences for explanation summary
        raw_preferences = []
        for score_result in scores_list:
            for dimension in DIMENSIONS:
                dim_result = getattr(score_result, dimension)
                raw_preferences.append(
                    {
                        "dimension": dimension,
                        "raw_preferences": dim_result.raw_preferences,
                    }
                )

        # Generate explanation summary
        explanation_summary = self._generate_explanation_summary_from_raw(
            raw_preferences
        )
        aggregated_metrics["explanation_summary"] = explanation_summary

        return aggregated_metrics


if __name__ == "__main__":
    # Test the evaluator with a simple example
    evaluator = DeepResearchPairwiseMetric()

    # Sample question and answers
    question = "What are the economic impacts of climate change?"
    baseline_answer = """
    Climate change has significant economic impacts globally:
    
    1. Agriculture: Changing weather patterns affect crop yields and food security.
    2. Infrastructure: Rising sea levels threaten coastal properties and infrastructure.
    3. Healthcare: Increased healthcare costs due to heat-related illnesses and vector-borne diseases.
    4. Energy: Changing demand patterns for heating and cooling.
    5. Insurance: Higher premiums due to increased natural disasters.
    """

    candidate_answer = """
    The economic impacts of climate change include:
    
    1. Agricultural disruption due to changing weather patterns and extreme events
    2. Infrastructure damage from rising sea levels and severe weather
    3. Increased healthcare costs
    4. Energy consumption shifts
    """

    print("Running evaluation...")
    result = evaluator.score(
        question=question,
        baseline_answer=baseline_answer,
        candidate_answer=candidate_answer,
    )

    print("\nScore Result:")
    print(result.model_dump_json(indent=2))

    # Print results
    print("\nEvaluation Results:")
    for dimension in DIMENSIONS:
        dimension_result = getattr(result, dimension)
        print(f"\n{dimension.upper()}:")
        print(f"Grade: {dimension_result.grade}")
        print(f"Score: {dimension_result.score:.2f}")
        print(f"Win: {dimension_result.is_win}")
        print(f"Tie: {dimension_result.is_tie}")

    # Aggregate results (with just one sample)
    aggregated = evaluator.aggregate([result])
    print("\nAggregated Metrics:")
    print(json.dumps(aggregated, indent=2))
    print(f"\nSupport: {aggregated['support']}")

    # Print overall
    print("\nOverall:")
    for metric, value in aggregated["overall"].items():
        if isinstance(value, float):
            print(f"{metric}: {value:.2f}")
        else:
            print(f"{metric}: {value}")

    print("\nExplanation Summary:")
    print(aggregated["explanation_summary"])

    print("\nTest complete.")
