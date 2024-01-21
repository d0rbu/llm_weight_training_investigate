import os
import torch as th
from core.weight_trajectory import PYTHIA_VARIANTS, get_pythia_weight_trajectory
from typing import Collection


def save_pythia_weights(
    variants: Collection[str] | None = None,
    output_dir: os.PathLike | str = "outputs",
) -> None:
    if variants is None:
        variants = PYTHIA_VARIANTS
    
    variants = set(variants)
    assert variants.issubset(PYTHIA_VARIANTS), f"Unknown variants: {PYTHIA_VARIANTS - variants}"

    for variant in variants:
        output_path = os.path.join(output_dir, f"pythia_{variant}_trajectory.pt")
        weight_trajectory = get_pythia_weight_trajectory(variant)
        th.save(weight_trajectory, output_path)

if __name__ == "__main__":
    save_pythia_weights()
