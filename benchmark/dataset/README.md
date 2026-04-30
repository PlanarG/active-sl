---
license: cc-by-4.0
language:
- en
pretty_name: Budget-Efficient Scaling Law Fitting Benchmark
task_categories:
- tabular-regression
tags:
- scaling-laws
- active-learning
- experimental-design
- benchmark
- language-modeling
size_categories:
- 1K<n<10K
dataset_info:
- config_name: data_constrained_scaling_law
  features:
  - name: group
    dtype: string
  - name: unique_tokens
    dtype: float64
  - name: params
    dtype: float64
  - name: tokens
    dtype: float64
  - name: loss
    dtype: float64
  splits:
  - name: train
    num_examples: 161
  - name: test
    num_examples: 21
- config_name: domain_mixture_scaling_law
  features:
  - name: group
    dtype: string
  - name: proportion_domain_1
    dtype: float64
  - name: proportion_domain_2
    dtype: float64
  - name: proportion_domain_3
    dtype: float64
  - name: proportion_domain_4
    dtype: float64
  - name: proportion_domain_5
    dtype: float64
  - name: loss_domain_1
    dtype: float64
  - name: loss_domain_2
    dtype: float64
  - name: loss_domain_3
    dtype: float64
  - name: loss_domain_4
    dtype: float64
  - name: loss_domain_5
    dtype: float64
  splits:
  - name: train
    num_examples: 80
  - name: test
    num_examples: 24
- config_name: farseer_scaling_law
  features:
  - name: group
    dtype: string
  - name: N
    dtype: float64
  - name: D
    dtype: float64
  - name: loss
    dtype: float64
  splits:
  - name: train
    num_examples: 404
  - name: test
    num_examples: 7
- config_name: lr_bsz_scaling_law
  features:
  - name: group
    dtype: string
  - name: lr
    dtype: float64
  - name: bsz
    dtype: float64
  - name: data_size
    dtype: float64
  - name: non_embedding_param_size
    dtype: float64
  - name: lm_loss
    dtype: float64
  splits:
  - name: train
    num_examples: 2702
  - name: test
    num_examples: 117
- config_name: moe_scaling_law
  features:
  - name: group
    dtype: string
  - name: num_experts
    dtype: float64
  - name: dense_parameter_count
    dtype: float64
  - name: loss_validation
    dtype: float64
  splits:
  - name: train
    num_examples: 193
  - name: test
    num_examples: 28
- config_name: parallel_scaling_law
  features:
  - name: num_params
    dtype: int64
  - name: parallel_size
    dtype: int64
  - name: group
    dtype: string
  - name: loss
    dtype: float64
  splits:
  - name: train
    num_examples: 36
  - name: test
    num_examples: 12
- config_name: sparsity_scaling_law
  features:
  - name: group
    dtype: string
  - name: P
    dtype: float64
  - name: N_active
    dtype: float64
  - name: N_dense
    dtype: float64
  - name: D1
    dtype: float64
  - name: D2
    dtype: float64
  - name: loss
    dtype: float64
  splits:
  - name: train
    num_examples: 70
  - name: test
    num_examples: 18
- config_name: vocab_scaling_law
  features:
  - name: group
    dtype: string
  - name: non_vocab_parameters
    dtype: float64
  - name: vocab_size
    dtype: float64
  - name: num_characters
    dtype: float64
  - name: unigram_normalized_loss
    dtype: float64
  splits:
  - name: train
    num_examples: 1080
  - name: test
    num_examples: 120
configs:
- config_name: data_constrained_scaling_law
  data_files:
  - split: train
    path: data_constrained_scaling_law/train-*
  - split: test
    path: data_constrained_scaling_law/test-*
- config_name: domain_mixture_scaling_law
  data_files:
  - split: train
    path: domain_mixture_scaling_law/train-*
  - split: test
    path: domain_mixture_scaling_law/test-*
- config_name: farseer_scaling_law
  data_files:
  - split: train
    path: farseer_scaling_law/train-*
  - split: test
    path: farseer_scaling_law/test-*
- config_name: lr_bsz_scaling_law
  data_files:
  - split: train
    path: lr_bsz_scaling_law/train-*
  - split: test
    path: lr_bsz_scaling_law/test-*
- config_name: moe_scaling_law
  data_files:
  - split: train
    path: moe_scaling_law/train-*
  - split: test
    path: moe_scaling_law/test-*
- config_name: parallel_scaling_law
  data_files:
  - split: train
    path: parallel_scaling_law/train-*
  - split: test
    path: parallel_scaling_law/test-*
- config_name: sparsity_scaling_law
  data_files:
  - split: train
    path: sparsity_scaling_law/train-*
  - split: test
    path: sparsity_scaling_law/test-*
- config_name: vocab_scaling_law
  data_files:
  - split: train
    path: vocab_scaling_law/train-*
  - split: test
    path: vocab_scaling_law/test-*
---

# Budget-Efficient Scaling Law Fitting Benchmark

This repository contains the scaling-law benchmark dataset used in
[Spend Less, Fit Better: Budget-Efficient Scaling Law Fitting via Active Experiment Selection](https://arxiv.org/abs/2604.22753).

The benchmark is designed for budget-aware sequential experimental design in scaling-law fitting. Each configuration provides a finite pool of candidate experiments, a held-out high-cost target region, task-specific covariates, observed outcomes, and companion scaling-law definitions in `laws.py`.

## Dataset Summary

The dataset contains 8 tabular regression tasks and 65 scaling-law instances. The tasks cover language-model scaling settings including pre-training hyperparameter tuning, data allocation, vocabulary design, domain mixture optimization, mixture-of-experts design, sparsity, parallel/inference-time scaling, and Farseer-style dense pre-training scaling.

Each task is stored as a separate Hugging Face configuration with `train` and `test` splits:

| Config | Train | Test | Feature columns | Target column(s) | Law instances |
|---|---:|---:|---|---|---:|
| `data_constrained_scaling_law` | 161 | 21 | `unique_tokens`, `params`, `tokens` | `loss` | 10 |
| `domain_mixture_scaling_law` | 80 | 24 | `proportion_domain_1` ... `proportion_domain_5` | `loss_domain_1` ... `loss_domain_5` | 10 |
| `farseer_scaling_law` | 404 | 7 | `N`, `D` | `loss` | 1 |
| `lr_bsz_scaling_law` | 2702 | 117 | `lr`, `bsz`, `data_size`, `non_embedding_param_size` | `lm_loss` | 10 |
| `moe_scaling_law` | 193 | 28 | `num_experts`, `dense_parameter_count` | `loss_validation` | 10 |
| `parallel_scaling_law` | 36 | 12 | `num_params`, `parallel_size` | `loss` | 10 |
| `sparsity_scaling_law` | 70 | 18 | `P`, `N_active` | `loss` | 4 |
| `vocab_scaling_law` | 1080 | 120 | `non_vocab_parameters`, `vocab_size`, `num_characters` | `unigram_normalized_loss` | 10 |

The `group` column identifies a task-specific subproblem or grouping. For example, domain-mixture rows are grouped by model scale, and parallel-scaling rows are grouped by evaluation corpus.

## Loading

```python
from datasets import load_dataset

ds = load_dataset("sijieli/scalebench", "lr_bsz_scaling_law")
print(ds)
print(ds["train"][0])
```

To load a local checkout before uploading:

```python
from datasets import load_dataset

ds = load_dataset(
    "parquet",
    data_files={
        "train": "lr_bsz_scaling_law/train-*.parquet",
        "test": "lr_bsz_scaling_law/test-*.parquet",
    },
)
```

## Intended Use

This benchmark is intended for evaluating experiment-selection and active experimental-design methods for scaling-law fitting under budget constraints. A typical episode treats the `train` split as the candidate pool of runnable experiments and the `test` split as the target region for extrapolation evaluation.

The benchmark can be used to compare methods that:

- choose experiments sequentially under a cost budget;
- fit nonlinear scaling laws from sparse observations;
- extrapolate to held-out high-cost regions;
- optimize target-region prediction quality rather than in-sample fit.

## Cost Proxies

The paper uses task-specific cost proxies to model heterogeneous experiment costs. The implementation in `registry.py` defines the default proxies:

| Config | Cost proxy |
|---|---|
| `data_constrained_scaling_law` | `6 * params * tokens` |
| `domain_mixture_scaling_law` | `1` |
| `farseer_scaling_law` | `6 * N * D` |
| `lr_bsz_scaling_law` | `6 * non_embedding_param_size * data_size` |
| `moe_scaling_law` | `dense_parameter_count * num_experts` |
| `parallel_scaling_law` | `num_params` |
| `sparsity_scaling_law` | `6 * N_dense * D1 + 6 * N_active * D2` |
| `vocab_scaling_law` | `non_vocab_parameters * num_characters` |

## Scaling-Law Definitions

Each task directory includes a `laws.py` file containing the parametric scaling-law families used in the benchmark. The functions are named `sl_1`, `sl_2`, etc., and each file exposes:

- `LAW_REGISTRY`: mapping from law ID to callable;
- `PARAM_COUNTS`: number of free parameters for each law;
- parameter bounds used by the fitting code.

These files are included to make the dataset self-contained for reproducing the benchmark protocol.

## Citation

If you use this benchmark, please cite:

```bibtex
@misc{li2026spendlessfitbetter,
  title={Spend Less, Fit Better: Budget-Efficient Scaling Law Fitting via Active Experiment Selection},
  author={Sijie Li and Shanda Li and Haowei Lin and Weiwei Sun and Ameet Talwalkar and Yiming Yang},
  year={2026},
  eprint={2604.22753},
  archivePrefix={arXiv},
  primaryClass={cs.LG},
  url={https://arxiv.org/abs/2604.22753}
}
```

## License

This dataset card follows the license metadata declared for this repository. Users should also respect the licenses and terms of the original data sources referenced by the paper.
