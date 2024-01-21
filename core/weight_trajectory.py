import torch as th


PYTHIA_VARIANTS = set(
    "70m",
    "160m",
    "410m",
    "1b",
    "1.4b",
    "2.8b",
    "6.9b",
    "12b",
)


def get_pythia_weight_trajectory(
    variant: str,
) -> None:
    variant_repo_name = f"EleutherAI/neox-ckpt-pythia-{variant}-deduped-v1"

    raise NotImplementedError("TODO: Load weights from variant_repo_name and save them in a single historical format")
