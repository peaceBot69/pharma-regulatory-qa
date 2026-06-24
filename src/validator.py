"""
validator.py
------------
Evaluate agent answer quality against ground-truth QA pairs.
Metrics: Exact Match, Token F1, ROUGE-L, Citation Hit Rate.

Ground-truth format (data/qa_pairs/qa_pairs.json):
[
  {
    "question": "What is the shelf life requirement per ICH Q1A?",
    "answer": "ICH Q1A requires a minimum of 12 months of stability data...",
    "sources": ["ich_q1a.pdf"]
  },
  ...
]

Usage:
    python src/validator.py --qa_file data/qa_pairs/qa_pairs.json
    python src/validator.py --qa_file data/qa_pairs/qa_pairs.json --sample 10
"""

import json
import logging
import argparse
import re
import string
from pathlib import Path
from dataclasses import dataclass, field

from rouge_score import rouge_scorer

from agent import PharmaQAAgent
from config import QA_PAIRS_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# RESULT SCHEMAS
# ─────────────────────────────────────────────

@dataclass
class SingleResult:
    question: str
    expected_answer: str
    predicted_answer: str
    expected_sources: list[str]
    predicted_sources: list[str]
    exact_match: bool
    token_f1: float
    rouge_l: float
    citation_hit: bool          # ≥1 expected source found in predicted sources


@dataclass
class EvalReport:
    total: int
    exact_match_rate: float
    avg_token_f1: float
    avg_rouge_l: float
    citation_hit_rate: float
    results: list[SingleResult] = field(default_factory=list)

    def print_summary(self):
        print("\n" + "="*60)
        print("VALIDATION REPORT")
        print("="*60)
        print(f"Total questions evaluated : {self.total}")
        print(f"Exact Match Rate          : {self.exact_match_rate:.1%}")
        print(f"Avg Token F1              : {self.avg_token_f1:.3f}")
        print(f"Avg ROUGE-L               : {self.avg_rouge_l:.3f}")
        print(f"Citation Hit Rate         : {self.citation_hit_rate:.1%}")
        print("="*60)

    def to_dict(self) -> dict:
        return {
            "total": self.total,
            "exact_match_rate": round(self.exact_match_rate, 4),
            "avg_token_f1": round(self.avg_token_f1, 4),
            "avg_rouge_l": round(self.avg_rouge_l, 4),
            "citation_hit_rate": round(self.citation_hit_rate, 4),
        }


# ─────────────────────────────────────────────
# METRIC FUNCTIONS
# ─────────────────────────────────────────────

def _normalise(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text


def exact_match(expected: str, predicted: str) -> bool:
    return _normalise(expected) == _normalise(predicted)


def token_f1(expected: str, predicted: str) -> float:
    """
    Token-level F1 between expected and predicted answer.
    Standard SQuAD-style metric.
    """
    exp_tokens = set(_normalise(expected).split())
    pred_tokens = set(_normalise(predicted).split())

    if not exp_tokens or not pred_tokens:
        return 0.0

    common = exp_tokens & pred_tokens
    if not common:
        return 0.0

    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(exp_tokens)
    f1 = 2 * precision * recall / (precision + recall)
    return round(f1, 4)


def rouge_l(expected: str, predicted: str) -> float:
    """ROUGE-L F1 score — measures longest common subsequence overlap."""
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    scores = scorer.score(expected, predicted)
    return round(scores["rougeL"].fmeasure, 4)


def citation_hit(expected_sources: list[str], predicted_sources: list[str]) -> bool:
    """
    Returns True if at least one expected source filename
    appears in the predicted sources list.
    """
    if not expected_sources:
        return True  # no citation requirement for this QA pair
    pred_lower = [s.lower() for s in predicted_sources]
    for exp in expected_sources:
        if exp.lower() in pred_lower:
            return True
    return False


# ─────────────────────────────────────────────
# QA LOADER
# ─────────────────────────────────────────────

def load_qa_pairs(qa_file: str) -> list[dict]:
    """Load ground-truth QA pairs from JSON file."""
    path = Path(qa_file)
    if not path.exists():
        raise FileNotFoundError(f"QA file not found: '{qa_file}'")
    with open(path, "r") as f:
        pairs = json.load(f)
    log.info(f"Loaded {len(pairs)} QA pairs from '{qa_file}'")
    return pairs


def create_sample_qa_file(output_path: str = "data/qa_pairs/qa_pairs.json"):
    """
    Creates a sample QA pairs JSON file with placeholder entries.
    Replace with real regulatory Q&A before running eval.
    """
    sample = [
        {
            "question": "What are the stability testing requirements per ICH Q1A?",
            "answer": "ICH Q1A requires long-term stability studies at 25°C/60% RH for 12 months minimum, with accelerated testing at 40°C/75% RH for 6 months.",
            "sources": ["ich_q1a.pdf"]
        },
        {
            "question": "What is the definition of a critical quality attribute (CQA)?",
            "answer": "A CQA is a physical, chemical, biological, or microbiological property or characteristic that should be within an appropriate limit, range, or distribution to ensure the desired product quality.",
            "sources": ["ich_q8.pdf"]
        },
        {
            "question": "What does FDA 21 CFR Part 11 govern?",
            "answer": "21 CFR Part 11 establishes criteria under which electronic records and electronic signatures are considered trustworthy, reliable, and equivalent to paper records and handwritten signatures.",
            "sources": ["21cfr_part11.pdf"]
        }
    ]
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(sample, f, indent=2)
    log.info(f"Sample QA file created at '{output_path}'")


# ─────────────────────────────────────────────
# EVALUATOR
# ─────────────────────────────────────────────

class Validator:
    def __init__(self):
        self.agent = PharmaQAAgent()

    def evaluate_single(self, qa_pair: dict) -> SingleResult:
        """Run agent on one QA pair and compute all metrics."""
        question = qa_pair["question"]
        expected_answer = qa_pair["answer"]
        expected_sources = qa_pair.get("sources", [])

        self.agent.reset_history()  # fresh session per question
        response = self.agent.ask(question)

        predicted_answer = response.answer
        predicted_sources = [s["source"] for s in response.sources]

        return SingleResult(
            question=question,
            expected_answer=expected_answer,
            predicted_answer=predicted_answer,
            expected_sources=expected_sources,
            predicted_sources=predicted_sources,
            exact_match=exact_match(expected_answer, predicted_answer),
            token_f1=token_f1(expected_answer, predicted_answer),
            rouge_l=rouge_l(expected_answer, predicted_answer),
            citation_hit=citation_hit(expected_sources, predicted_sources),
        )

    def evaluate(self, qa_pairs: list[dict], sample: int = None) -> EvalReport:
        """
        Run eval across all (or sampled) QA pairs.
        Returns EvalReport with per-question results + aggregate metrics.
        """
        if sample:
            import random
            random.seed(42)
            qa_pairs = random.sample(qa_pairs, min(sample, len(qa_pairs)))
            log.info(f"Sampled {len(qa_pairs)} pairs for evaluation.")

        results = []
        for i, pair in enumerate(qa_pairs, 1):
            log.info(f"Evaluating [{i}/{len(qa_pairs)}]: {pair['question'][:60]}...")
            try:
                result = self.evaluate_single(pair)
                results.append(result)
                log.info(
                    f"  EM={result.exact_match} | "
                    f"F1={result.token_f1} | "
                    f"ROUGE-L={result.rouge_l} | "
                    f"CitHit={result.citation_hit}"
                )
            except Exception as e:
                log.error(f"  Failed: {e}")
                continue

        if not results:
            raise RuntimeError("No results produced. Check agent + QA file.")

        n = len(results)
        report = EvalReport(
            total=n,
            exact_match_rate=sum(r.exact_match for r in results) / n,
            avg_token_f1=sum(r.token_f1 for r in results) / n,
            avg_rouge_l=sum(r.rouge_l for r in results) / n,
            citation_hit_rate=sum(r.citation_hit for r in results) / n,
            results=results,
        )
        return report

    def save_report(self, report: EvalReport, output_path: str = "data/qa_pairs/eval_report.json"):
        """Save full report to JSON."""
        data = report.to_dict()
        data["per_question"] = [
            {
                "question": r.question,
                "exact_match": r.exact_match,
                "token_f1": r.token_f1,
                "rouge_l": r.rouge_l,
                "citation_hit": r.citation_hit,
                "predicted_sources": r.predicted_sources,
            }
            for r in report.results
        ]
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)
        log.info(f"Report saved to '{output_path}'")


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pharma Doc QA — Validator")
    parser.add_argument("--qa_file", default="data/qa_pairs/qa_pairs.json")
    parser.add_argument("--sample", type=int, default=None,
                        help="Evaluate on N random pairs (default: all)")
    parser.add_argument("--create_sample", action="store_true",
                        help="Create a sample QA file and exit")
    parser.add_argument("--save_report", action="store_true",
                        help="Save eval report to JSON")
    args = parser.parse_args()

    if args.create_sample:
        create_sample_qa_file()
        exit(0)

    qa_pairs = load_qa_pairs(args.qa_file)
    validator = Validator()
    report = validator.evaluate(qa_pairs, sample=args.sample)
    report.print_summary()

    if args.save_report:
        validator.save_report(report)
