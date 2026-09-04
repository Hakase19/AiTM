"""Download complete datasets for AiTM experiments.

Paper Section 4.1: MMLU (biology/physics), HumanEval (164 samples)
"""

import json
import os
from typing import List, Dict, Any


# MMLU Biology subtasks (from MMLU paper)
MMLU_BIOLOGY_TASKS = [
    "abstract_algebra",
    "anatomy",
    "astronomy",
    "college_biology",
    "college_chemistry",
    "conceptual_physics",
    "elementary_mathematics",
    "high_school_biology",
    "high_school_chemistry",
    "high_school_physics",
    "medical_genetics",
    "nutrition",
    "professional_medicine",
    "virology",
]

# MMLU Physics subtasks
MMLU_PHYSICS_TASKS = [
    "conceptual_physics",
    "high_school_physics",
    "college_physics",
]


def download_mmlu_task(task_name: str, output_dir: str) -> List[Dict]:
    """Download a single MMLU task."""
    from datasets import load_dataset

    print(f"  Downloading {task_name}...")
    try:
        dataset = load_dataset("cais/mmlu", task_name, split="test", trust_remote_code=True)
        samples = []
        for item in dataset:
            question = item.get("question", "")
            choices = item.get("choices", [])
            answer_idx = item.get("answer", 0)

            # Format as multiple choice
            formatted = question + "\n"
            for j, choice in enumerate(choices):
                formatted += f"{chr(65+j)}. {choice}\n"

            samples.append({
                "task": task_name,
                "question": question,
                "choices": choices,
                "answer": chr(65 + answer_idx),
                "formatted": formatted,
            })

        # Save
        output_path = os.path.join(output_dir, f"{task_name}.json")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(samples, f, indent=2, ensure_ascii=False)

        print(f"    Saved {len(samples)} samples")
        return samples

    except Exception as e:
        print(f"    Error: {e}")
        return []


def download_humaneval(output_dir: str) -> List[Dict]:
    """Download complete HumanEval dataset (164 samples)."""
    from datasets import load_dataset

    print("Downloading HumanEval (164 samples)...")
    try:
        # Try evalplus first
        dataset = load_dataset("evalplus/humanevalplus", split="test", trust_remote_code=True)
        samples = []
        for i, item in enumerate(dataset):
            samples.append({
                "task_id": item.get("task_id", f"HumanEval/{i}"),
                "prompt": item.get("prompt", ""),
                "canonical_solution": item.get("canonical_solution", ""),
                "test": item.get("test", ""),
                "entry_point": item.get("entry_point", ""),
            })

        output_path = os.path.join(output_dir, "humaneval.json")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(samples, f, indent=2, ensure_ascii=False)

        print(f"  Saved {len(samples)} samples")
        return samples

    except Exception as e:
        print(f"  Error: {e}")
        return []


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # Download MMLU Biology
    print("=" * 50)
    print("Downloading MMLU Biology")
    print("=" * 50)
    bio_dir = os.path.join(base_dir, "mmlu_biology")
    os.makedirs(bio_dir, exist_ok=True)
    for task in MMLU_BIOLOGY_TASKS:
        download_mmlu_task(task, bio_dir)

    # Download MMLU Physics
    print("\n" + "=" * 50)
    print("Downloading MMLU Physics")
    print("=" * 50)
    phy_dir = os.path.join(base_dir, "mmlu_physics")
    os.makedirs(phy_dir, exist_ok=True)
    for task in MMLU_PHYSICS_TASKS:
        download_mmlu_task(task, phy_dir)

    # Download HumanEval
    print("\n" + "=" * 50)
    print("Downloading HumanEval")
    print("=" * 50)
    he_dir = os.path.join(base_dir, "humaneval")
    os.makedirs(he_dir, exist_ok=True)
    download_humaneval(he_dir)

    print("\n" + "=" * 50)
    print("All datasets downloaded!")
    print("=" * 50)


if __name__ == "__main__":
    main()
