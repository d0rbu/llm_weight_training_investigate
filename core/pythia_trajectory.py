import tempfile
import os
import torch as th
from core.util import vectorize_model
from transformers import GPTNeoXForCausalLM


PYTHIA_VARIANTS = set(
    "14m",
    "31m",
    "70m",
    "160m",
    "410m",
    "1b",
    "1.4b",
    "2.8b",
    "6.9b",
    "12b",
)
PYTHIA_STEPS = [2 ** i for i in range(10)]  # 1 to 512
PYTHIA_STEPS.insert(0, 0)  # 0 is the initial checkpoint
PYTHIA_STEPS.append(1000 * i for i in range(1, 144))  # 1k to 144k

def pythia_variant_to_name(variant: str) -> str:
    return f"EleutherAI/pythia-{variant}-deduped"

def save_pythia_weight_trajectory(
    variant: str,
    output_dir: os.PathLike | str = "outputs",
) -> None:
    assert variant in PYTHIA_VARIANTS, f"Unknown variant: {variant}"

    variant_name = pythia_variant_to_name(variant)

    with th.no_grad():
        for step in PYTHIA_STEPS:
            print(f"Loading {variant_name} at step {step}")
            with tempfile.TemporaryDirectory() as temp_dir:
                model = GPTNeoXForCausalLM.from_pretrained(
                    variant_name,
                    revision=f"step{step}",
                    cache_dir=temp_dir,
                )

                weights = vectorize_model(model)
                del model

                th.save(weights, os.path.join(output_dir, variant_name, f"{step}.pt"))
