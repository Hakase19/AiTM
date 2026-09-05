"""Dataset loader for AiTM experiments.

Paper Section 4.1: MMLU (biology/physics), HumanEval (164 samples)
"""

import json
import os
from typing import List, Dict, Any


def load_mmlu(domain: str = "biology") -> List[Dict[str, Any]]:
    """Load all MMLU samples for a domain.

    Args:
        domain: 'biology' or 'physics'

    Returns:
        List of all samples across subtasks.
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, f"mmlu_{domain}")

    all_samples = []
    for filename in sorted(os.listdir(data_dir)):
        if filename.endswith(".json"):
            filepath = os.path.join(data_dir, filename)
            with open(filepath, "r", encoding="utf-8") as f:
                samples = json.load(f)
                all_samples.extend(samples)

    return all_samples


def load_humaneval() -> List[Dict[str, Any]]:
    """Load all HumanEval samples (164 total).

    Returns:
        List of all HumanEval samples.
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    filepath = os.path.join(base_dir, "humaneval", "test.json")

    with open(filepath, "r", encoding="utf-8") as f:
        samples = json.load(f)

    return samples


def load_mbpp() -> List[Dict[str, Any]]:
    """Load all MBPP samples (974 total).

    Returns:
        List of all MBPP samples.
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    filepath = os.path.join(base_dir, "mbpp", "test.json")
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def get_dataset_stats():
    """Print statistics for all datasets."""
    stats = {}

    # MMLU Biology
    bio_samples = load_mmlu("biology")
    stats["mmlu_biology"] = len(bio_samples)

    # MMLU Physics
    phy_samples = load_mmlu("physics")
    stats["mmlu_physics"] = len(phy_samples)

    # HumanEval
    he_samples = load_humaneval()
    stats["humaneval"] = len(he_samples)

    # MBPP
    mbpp_samples = load_mbpp()
    stats["mbpp"] = len(mbpp_samples)

    return stats


if __name__ == "__main__":
    stats = get_dataset_stats()
    print("Dataset Statistics:")
    for name, count in stats.items():
        print(f"  {name}: {count} samples")
