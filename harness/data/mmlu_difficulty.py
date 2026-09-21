"""
Load 500 MMLU prompts from HuggingFace, map categories to difficulty tiers,
save to bench/data/mmlu_500.jsonl.

Usage:
    uv run bench/data/mmlu_difficulty.py
"""

import json
import random
from pathlib import Path

from datasets import load_dataset

from smartroute.classifier import DifficultyTier

# MMLU category → difficulty tier mapping
# Based on published model accuracy tables (GPT-4 era benchmarks)
CATEGORY_TIER: dict[str, DifficultyTier] = {
    # EASY — high accuracy across most models (>80%)
    "elementary_mathematics": DifficultyTier.EASY,
    "world_religions": DifficultyTier.EASY,
    "nutrition": DifficultyTier.EASY,
    "high_school_geography": DifficultyTier.EASY,
    "high_school_us_history": DifficultyTier.EASY,
    "sociology": DifficultyTier.EASY,
    "prehistory": DifficultyTier.EASY,
    "human_sexuality": DifficultyTier.EASY,
    "us_foreign_policy": DifficultyTier.EASY,
    "global_facts": DifficultyTier.EASY,
    "public_relations": DifficultyTier.EASY,
    "security_studies": DifficultyTier.EASY,
    "management": DifficultyTier.EASY,
    "miscellaneous": DifficultyTier.EASY,
    "moral_scenarios": DifficultyTier.EASY,
    "high_school_world_history": DifficultyTier.EASY,
    "medical_genetics": DifficultyTier.EASY,
    "clinical_knowledge": DifficultyTier.EASY,
    "human_aging": DifficultyTier.EASY,
    "international_law": DifficultyTier.EASY,
    "jurisprudence": DifficultyTier.EASY,
    "virology": DifficultyTier.EASY,
    # MEDIUM — moderate difficulty, models 60-80% accuracy
    "high_school_biology": DifficultyTier.MEDIUM,
    "high_school_chemistry": DifficultyTier.MEDIUM,
    "high_school_physics": DifficultyTier.MEDIUM,
    "high_school_mathematics": DifficultyTier.MEDIUM,
    "high_school_statistics": DifficultyTier.MEDIUM,
    "high_school_macroeconomics": DifficultyTier.MEDIUM,
    "high_school_microeconomics": DifficultyTier.MEDIUM,
    "high_school_psychology": DifficultyTier.MEDIUM,
    "high_school_european_history": DifficultyTier.MEDIUM,
    "high_school_government_and_politics": DifficultyTier.MEDIUM,
    "high_school_computer_science": DifficultyTier.MEDIUM,
    "college_biology": DifficultyTier.MEDIUM,
    "college_chemistry": DifficultyTier.MEDIUM,
    "college_computer_science": DifficultyTier.MEDIUM,
    "college_medicine": DifficultyTier.MEDIUM,
    "marketing": DifficultyTier.MEDIUM,
    "econometrics": DifficultyTier.MEDIUM,
    "professional_accounting": DifficultyTier.MEDIUM,
    "professional_psychology": DifficultyTier.MEDIUM,
    "business_ethics": DifficultyTier.MEDIUM,
    "logical_fallacies": DifficultyTier.MEDIUM,
    "philosophy": DifficultyTier.MEDIUM,
    "anatomy": DifficultyTier.MEDIUM,
    "astronomy": DifficultyTier.MEDIUM,
    "conceptual_physics": DifficultyTier.MEDIUM,
    "computer_security": DifficultyTier.MEDIUM,
    "machine_learning": DifficultyTier.MEDIUM,
    # HARD — challenging, models typically <60% accuracy
    "abstract_algebra": DifficultyTier.HARD,
    "formal_logic": DifficultyTier.HARD,
    "professional_law": DifficultyTier.HARD,
    "professional_medicine": DifficultyTier.HARD,
    "moral_disputes": DifficultyTier.HARD,
    "college_mathematics": DifficultyTier.HARD,
    "electrical_engineering": DifficultyTier.HARD,
    "college_physics": DifficultyTier.HARD,
    "mathematical_reasoning": DifficultyTier.HARD,
}

CHOICES_LABELS = ["A", "B", "C", "D"]


def format_prompt(question: str, choices: list[str]) -> str:
    """Format MMLU question as multiple-choice prompt."""
    lines = [question]
    for label, choice in zip(CHOICES_LABELS, choices):
        lines.append(f"{label}. {choice}")
    return "\n".join(lines)


def get_tier(subject: str) -> DifficultyTier:
    """Map MMLU subject to difficulty tier; default MEDIUM for unmapped."""
    return CATEGORY_TIER.get(subject, DifficultyTier.MEDIUM)


def load_mmlu_500(seed: int = 42, n: int = 500) -> list[dict]:
    """
    Load MMLU test split, deduplicate, sample n prompts preserving natural distribution.
    Returns list of dicts ready for JSONL serialisation.
    """
    print("Loading cais/mmlu (all subjects, test split)...")
    dataset = load_dataset("cais/mmlu", name="all", split="test", trust_remote_code=True)

    # Deduplicate by (question, answer)
    seen: set[tuple[str, int]] = set()
    rows = []
    for row in dataset:
        key = (row["question"], row["answer"])
        if key not in seen:
            seen.add(key)
            rows.append(row)

    print(f"Unique rows after dedup: {len(rows)}")

    # Sample without replacement, preserving natural distribution
    rng = random.Random(seed)
    sampled = rng.sample(rows, min(n, len(rows)))

    records = []
    for row in sampled:
        subject: str = row["subject"]
        tier = get_tier(subject)
        choices: list[str] = row["choices"]
        correct_idx: int = row["answer"]
        correct_label = CHOICES_LABELS[correct_idx]

        records.append(
            {
                "source_id": f"mmlu-{row['subject']}-{hash((row['question'], row['answer'])) & 0xFFFFFFFF:08x}",
                "subject": subject,
                "difficulty_tier": tier.value,
                "prompt": format_prompt(row["question"], choices),
                "correct_answer": correct_label,
            }
        )

    return records


def print_distribution(records: list[dict]) -> None:
    from collections import Counter

    tier_counts = Counter(r["difficulty_tier"] for r in records)
    subject_counts = Counter(r["subject"] for r in records)

    print(f"\nTotal prompts: {len(records)}")
    print("\nTier distribution:")
    for tier in ["EASY", "MEDIUM", "HARD"]:
        count = tier_counts.get(tier, 0)
        pct = count / len(records) * 100
        print(f"  {tier.upper():6s}: {count:4d} ({pct:.1f}%)")

    print("\nTop 10 subjects:")
    for subject, count in subject_counts.most_common(10):
        tier = get_tier(subject).value
        print(f"  {subject:<40s} {count:3d}  [{tier}]")


def main() -> None:
    out_path = Path(__file__).parent / "mmlu_500.jsonl"

    records = load_mmlu_500(seed=42, n=500)
    print_distribution(records)

    with out_path.open("w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")

    print(f"\nSaved {len(records)} records to {out_path}")

    # Spot-check
    print("\nSample prompts:")
    for record in records[:2]:
        print(f"\n  [{record['difficulty_tier'].upper()}] {record['subject']}")
        print(f"  source_id: {record['source_id']}")
        print(f"  correct: {record['correct_answer']}")
        prompt_preview = record["prompt"][:120].replace("\n", " | ")
        print(f"  prompt: {prompt_preview}...")


if __name__ == "__main__":
    main()
