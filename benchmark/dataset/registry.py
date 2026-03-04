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


def _cost_sft(row: dict) -> float:
    return float(row["sft_data_size"])


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
    ),
    "sft_scaling_law": DatasetInfo(
        name="sft_scaling_law",
        feature_cols=["sft_data_size"],
        target_cols=["sft_loss"],
        cost_fn=_cost_sft,
    ),
}


def get_dataset_info(name: str) -> DatasetInfo:
    if name not in DATASET_REGISTRY:
        raise KeyError(f"Unknown dataset: {name}. Available: {list(DATASET_REGISTRY)}")
    return DATASET_REGISTRY[name]
