import os
import torch as th
from core.pythia_trajectory import PYTHIA_VARIANTS, save_pythia_weight_trajectory, pythia_variant_to_name
from typing import Collection


def save_pythia_weights(
    variants: Collection[str] | None = None,
    output_dir: os.PathLike | str = "outputs",
) -> None:
    if variants is None:
        variants = PYTHIA_VARIANTS

    for variant in variants:
        save_pythia_weight_trajectory(variant, variant)

if __name__ == "__main__":
    save_pythia_weights()
