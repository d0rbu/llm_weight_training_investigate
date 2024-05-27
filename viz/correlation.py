import os
import argparse
import torch as th
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from core.pythia_trajectory import PYTHIA_VARIANTS, pythia_variant_to_name
from core.util import group_weights_by_layer
from tqdm import tqdm
from typing import Collection
from celluloid import Camera


def color_from_name(name: str) -> str | tuple[float, float, float]:
    if "attention.query_key_value" in name:
        return "brown"

    if "attention.dense" in name:
        return "peru"

    if "mlp.dense_4h_to_h" in name:
        return "gold"

    if "mlp.dense_h_to_4h" in name:
        return "yellowgreen"

    if "input_layernorm" in name:
        return "forestgreen"

    if "post_attention_layernorm" in name:
        return "darkcyan"

    if "final_layer_norm" in name:
        return "slategrey"

    if "embed_in" in name:
        return "slateblue"
    
    if "embed_out" in name:
        return "hotpink"

    return "black"


def visualize_trajectories(
    variants: Collection[str] | None = None,
    trajectory_dir: os.PathLike | str = "outputs",
    output_dir: os.PathLike | str = "viz_outputs",
    random_weight_subset: int | float | None = None,
    num_steps: int | None = None,
) -> None:
    if variants is None:
        variants = PYTHIA_VARIANTS
    
    if random_weight_subset is None:
        random_weight_subset = 1.0
    
    if random_weight_subset > 1.0:
        random_weight_subset = int(random_weight_subset)

    for variant in variants:
        variant_name = pythia_variant_to_name(variant)
        model_dir = os.path.join(trajectory_dir, variant_name)
        files = os.listdir(model_dir)
        # sort files by filename (not including the extension) to ensure the order is correct
        files.sort(key=lambda x: int(x.split(".")[0]))

        print(f"Loading final weights {files[-1]}")
        final_weights, final_weight_sizes = th.load(os.path.join(model_dir, files[-1]))

        num_params = random_weight_subset if isinstance(random_weight_subset, int) else int(random_weight_subset * final_weights.shape[0])
        random_weights = th.randperm(final_weights.shape[0])[:num_params]
        final_weights_by_layer = group_weights_by_layer(final_weights, final_weight_sizes, random_weights)

        camera = Camera(plt.figure())
        plt.xlim(-2, 2)
        plt.ylim(-2, 2)

        for i, file in enumerate(tqdm(files, total=len(files), desc=f"Visualizing {variant}")):
            if num_steps is not None and i >= num_steps:
                break

            step_num = int(file.split(".")[0])
            weights, weight_sizes = th.load(os.path.join(model_dir, file))

            weights_by_layer = group_weights_by_layer(weights, weight_sizes, random_weights)

            # measure the correlation between the final weights and the current weights
            correlation = th.corrcoef(th.stack([weights, final_weights]))[0, 1].item()

            # color each data point based on the layer -> color map
            for layer_name, layer_weights in weights_by_layer.items():
                final_layer_weights = final_weights_by_layer[layer_name]
                plt.scatter(layer_weights, final_layer_weights, color=color_from_name(layer_name))
            plt.legend([f"Step {step_num}\nCorrelation: {correlation:.2f}"], loc="upper right")
            camera.snap()

            del weights  # free up memory after plotting

        print("Animating")
        anim = camera.animate()

        print("Saving")
        save_path = os.path.join(output_dir, f"{variant_name}.mp4")
        save_dir = os.path.dirname(save_path)
        os.makedirs(save_dir, exist_ok=True)
        anim.save(save_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize Pythia weight trajectories")
    parser.add_argument("--variants", type=str, nargs="+", default=["14m", "31m"], help="Pythia variants to visualize")
    parser.add_argument("--trajectory_dir", type=str, default="outputs", help="Directory trajectory weights are saved to")
    parser.add_argument("--output_dir", type=str, default="viz_outputs", help="Directory to save visualizations")
    parser.add_argument("--random_weight_subset", type=float, default=None, help="Fraction of weights to visualize or integer number of weights to visualize")
    parser.add_argument("--num_steps", type=int, default=None, help="Number of steps to visualize")
    args = parser.parse_args()

    visualize_trajectories(
        args.variants,
        args.trajectory_dir,
        args.output_dir,
        args.random_weight_subset,
        args.num_steps,
    )