import os
import argparse
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
        save_pythia_weight_trajectory(variant, output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Save Pythia weights")
    parser.add_argument("--variants", type=str, nargs="+", default=["14m", "31m"], help="Pythia variants to save")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Directory to save weights")
    args = parser.parse_args()

    save_pythia_weights(
        args.variants,
        args.output_dir,
    )
