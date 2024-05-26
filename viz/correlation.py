import os
import torch as th
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from core.pythia_trajectory import PYTHIA_VARIANTS, pythia_variant_to_name
from typing import Collection


def visualize_trajectories(
    variants: Collection[str] | None = None,
    trajectory_dir: os.PathLike | str = "outputs",
) -> None:
    if variants is None:
        variants = PYTHIA_VARIANTS

    for variant in variants:
        print(f"Visualizing {variant}")
        variant_name = pythia_variant_to_name(variant)
        model_dir = os.path.join(trajectory_dir, variant_name)
        files = sorted(os.listdir(model_dir))
        artists = []

        final_weights = th.load(os.path.join(model_dir, files[-1]))

        for file in files:
            print(f"Loading {file}")
            weights = th.load(os.path.join(model_dir, file))
            artists.append(plt.scatter(weights, final_weights))

        print("Animating")
        ani = animation.ArtistAnimation(
            plt.figure(),
            artists,
            interval=50,
            blit=True,
            repeat_delay=1000,
        )

        print("Saving")
        ani.save(f"{variant_name}.mp4")

if __name__ == "__main__":
    visualize_trajectories()