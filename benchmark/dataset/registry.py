"""Dataset registry: metadata, feature/target columns, and cost functions."""

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np


@dataclass
class DatasetInfo:
    name: str
    feature_cols: List[str]
    target_cols: List[str]
    group_col: str = "group"
    cost_fn: Optional[Callable[[dict], float]] = None
    # Extra columns needed for cost but not used as features
    cost_extra_cols: List[str] = field(default_factory=list)
    # Per-dataset budget checkpoints (overrides runner default if set)
    budget_checkpoints: Optional[List[float]] = None


def _cost_data_constrained(row: dict) -> float:
    return 6.0 * row["params"] * row["tokens"]


def _cost_parallel(row: dict) -> float:
    return float(row["num_params"])


def _cost_moe(row: dict) -> float:
    return float(row["dense_parameter_count"]) * float(row["num_experts"])


def _cost_easy_question(row: dict) -> float:
    return float(row["flops"])


def _cost_vocab(row: dict) -> float:
    return float(row["non_vocab_parameters"]) * float(row["num_characters"])


def _cost_lr_bsz(row: dict) -> float:
    return 6.0 * float(row["non_embedding_param_size"]) * float(row["data_size"])


def _cost_domain_mixture(row: dict) -> float:
    return 1.0


def _cost_chinchilla(row: dict) -> float:
    return 6.0 * float(row["N"]) * float(row["D"])


def _cost_farseer(row: dict) -> float:
    return 6.0 * float(row["N"]) * float(row["D"])


def _cost_sft(row: dict) -> float:
    return float(row["sft_data_size"])


def _cost_sae(row: dict) -> float:
    return float(row["n"]) ** 1.6


def _cost_distillation(row: dict) -> float:
    return 6.0 * float(row["NS"]) * float(row["DS"])


def _cost_sparsity(row: dict) -> float:
    return 6.0 * row["N_dense"] * row["D1"] + 6.0 * row["N_active"] * row["D2"]


DATASET_REGISTRY: Dict[str, DatasetInfo] = {
    "data_constrained_scaling_law": DatasetInfo(
        name="data_constrained_scaling_law",
        feature_cols=["unique_tokens", "params", "tokens"],
        target_cols=["loss"],
        cost_fn=_cost_data_constrained,
    ),
    "parallel_scaling_law": DatasetInfo(
        name="parallel_scaling_law",
        feature_cols=["num_params", "parallel_size"],
        target_cols=["loss"],
        cost_fn=_cost_parallel,
    ),
    "moe_scaling_law": DatasetInfo(
        name="moe_scaling_law",
        feature_cols=["num_experts", "dense_parameter_count"],
        target_cols=["loss_validation"],
        cost_fn=_cost_moe,
    ),
    "easy_question_scaling_law": DatasetInfo(
        name="easy_question_scaling_law",
        feature_cols=["log_flops"],
        target_cols=["brier_score"],
        cost_fn=_cost_easy_question,
        cost_extra_cols=["flops"],
        budget_checkpoints=[0.01, 0.05, 0.1]
    ),
    "vocab_scaling_law": DatasetInfo(
        name="vocab_scaling_law",
        feature_cols=["non_vocab_parameters", "vocab_size", "num_characters"],
        target_cols=["unigram_normalized_loss"],
        cost_fn=_cost_vocab,
    ),
    "lr_bsz_scaling_law": DatasetInfo(
        name="lr_bsz_scaling_law",
        feature_cols=["lr", "bsz", "data_size", "non_embedding_param_size"],
        target_cols=["lm_loss"],
        cost_fn=_cost_lr_bsz,
    ),
    "domain_mixture_scaling_law": DatasetInfo(
        name="domain_mixture_scaling_law",
        feature_cols=[
            "proportion_domain_1",
            "proportion_domain_2",
            "proportion_domain_3",
            "proportion_domain_4",
            "proportion_domain_5",
        ],
        target_cols=[
            "loss_domain_1",
            "loss_domain_2",
            "loss_domain_3",
            "loss_domain_4",
            "loss_domain_5",
        ],
        cost_fn=_cost_domain_mixture,
        budget_checkpoints=[0.2, 0.35, 0.5],
    ),
    "chinchilla_scaling_law": DatasetInfo(
        name="chinchilla_scaling_law",
        feature_cols=["N", "D"],
        target_cols=["loss"],
        cost_fn=_cost_chinchilla,
    ),
    "farseer_scaling_law": DatasetInfo(
        name="farseer_scaling_law",
        feature_cols=["N", "D"],
        target_cols=["loss"],
        cost_fn=_cost_farseer,
    ),
    "sft_scaling_law": DatasetInfo(
        name="sft_scaling_law",
        feature_cols=["sft_data_size"],
        target_cols=["sft_loss"],
        cost_fn=_cost_sft,
    ),
    "sae_scaling_law": DatasetInfo(
        name="sae_scaling_law",
        feature_cols=["n", "k"],
        target_cols=["loss"],
        cost_fn=_cost_sae,
    ),
    "distillation_scaling_law": DatasetInfo(
        name="distillation_scaling_law",
        feature_cols=["NS", "DS", "LT"],
        target_cols=["LS"],
        cost_fn=_cost_distillation,
    ),
    "sparsity_scaling_law": DatasetInfo(
        name="sparsity_scaling_law",
        feature_cols=["P", "N_active"],
        target_cols=["loss"],
        cost_fn=_cost_sparsity,
        cost_extra_cols=["N_dense", "D1", "D2"],
        budget_checkpoints=[0.2, 0.35, 0.5]
    ),
}


def get_dataset_info(name: str) -> DatasetInfo:
    if name not in DATASET_REGISTRY:
        raise KeyError(f"Unknown dataset: {name}. Available: {list(DATASET_REGISTRY)}")
    return DATASET_REGISTRY[name]
