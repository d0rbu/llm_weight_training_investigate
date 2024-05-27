import torch as th
import torch.nn as nn


def vectorize_model(model: nn.Module) -> tuple[th.Tensor, list[tuple[str, int]]]:
    """
    Vectorize an nn.Module into a single tensor of weights.
    Also returns a list of (layer_name, num_params) tuples ordered alphabetically by layer_name, same as the weights.
    """
    state_dict = model.state_dict()
    ordered_keys = sorted(state_dict.keys())
    weights = []
    layer_sizes = []

    for key in ordered_keys:
        flattened_weights = state_dict[key].flatten().cpu()
        weights.append(flattened_weights)
        layer_sizes.append((key, flattened_weights.shape[0]))

    return th.cat(weights), layer_sizes

def generate_layer_index_map(layer_sizes: list[tuple[str, int]]) -> th.Tensor:
    """
    Create a map from layer indices to the index in the weight sizes list that corresponds to that weight.
    """
    layer_index_map = th.zeros(sum(size for _, size in layer_sizes), dtype=th.int64)
    start = 0

    for i, (_, size) in enumerate(layer_sizes):
        layer_index_map[start:start + size] = i
        start += size

    return layer_index_map

def group_weights_by_layer(weights: th.Tensor, weight_sizes: list[tuple[str, int]], weight_mask: th.Tensor | None = None) -> dict[str, th.Tensor]:
    """
    Group weights by layer name.
    """
    weight_map = generate_layer_index_map(weight_sizes)  # weight to layer index map
    if weight_mask is not None:
        weights = weights[weight_mask]
        weight_map = weight_map[weight_mask]

    weights_by_layer = {
        layer_name: weights[weight_map == layer_idx]
        for layer_idx, (layer_name, _) in enumerate(weight_sizes)
    }

    return weights_by_layer
