import os
import torch as th
from core.weight_trajectory import PYTHIA_VARIANTS
from typing import Collection


def visualize_trajectories(
    variants: Collection[str] | None = None,
    trajectory_dir: os.PathLike | str = "outputs",
) -> None:
    if variants is None:
        variants = PYTHIA_VARIANTS

    for variant in variants:
        trajectory_path = os.path.join(trajectory_dir, f"pythia_{variant}_trajectory.pt")
        trajectory = th.load(trajectory_path)

        raise NotImplementedError("TODO: Visualize trajectory")

if __name__ == "__main__":
    visualize_trajectories()