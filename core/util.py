import torch as th
import torch.nn as nn


def vectorize_model(model: nn.Module) -> th.Tensor:
    """Vectorize an nn.Module into a single tensor of weights."""
    state_dict = model.state_dict()
    ordered_keys = sorted(state_dict.keys())
    weights = []

    for key in ordered_keys:
        weights.append(state_dict[key].flatten().cpu())

    return th.cat(weights)