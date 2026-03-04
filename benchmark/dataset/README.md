---
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
    num_bytes: 7084
    num_examples: 161
  - name: test
    num_bytes: 924
    num_examples: 21
  download_size: 7325
  dataset_size: 8008
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
    num_bytes: 7020
    num_examples: 80
  - name: test
    num_bytes: 2106
    num_examples: 24
  download_size: 15342
  dataset_size: 9126
- config_name: easy_question_scaling_law
  features:
  - name: group
    dtype: string
  - name: log_flops
    dtype: float64
  - name: flops
    dtype: float64
  - name: brier_score
    dtype: float64
  - name: model_name
    dtype: string
  splits:
  - name: train
    num_bytes: 27304
    num_examples: 389
  - name: test
    num_bytes: 8530
    num_examples: 127
  download_size: 13707
  dataset_size: 35834
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
    num_bytes: 140504.0
    num_examples: 2702
  - name: test
    num_bytes: 6084.0
    num_examples: 117
  download_size: 39110
  dataset_size: 146588.0
- config_name: lr_bsz_scaling_law_modified
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
    num_bytes: 186836
    num_examples: 3593
  - name: test
    num_bytes: 6084
    num_examples: 117
  download_size: 44596
  dataset_size: 192920
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
    num_bytes: 6948
    num_examples: 193
  - name: test
    num_bytes: 1008
    num_examples: 28
  download_size: 6463
  dataset_size: 7956
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
    num_bytes: 1170
    num_examples: 36
  - name: test
    num_bytes: 390
    num_examples: 12
  download_size: 4551
  dataset_size: 1560
- config_name: sft_scaling_law
  features:
  - name: group
    dtype: string
  - name: sft_data_size
    dtype: int64
  - name: sft_loss
    dtype: float64
  splits:
  - name: train
    num_bytes: 25896
    num_examples: 504
  - name: test
    num_bytes: 2158
    num_examples: 42
  download_size: 9274
  dataset_size: 28054
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
    num_bytes: 47520
    num_examples: 1080
  - name: test
    num_bytes: 5280
    num_examples: 120
  download_size: 23841
  dataset_size: 52800
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
- config_name: easy_question_scaling_law
  data_files:
  - split: train
    path: easy_question_scaling_law/train-*
  - split: test
    path: easy_question_scaling_law/test-*
- config_name: lr_bsz_scaling_law
  data_files:
  - split: train
    path: lr_bsz_scaling_law/train-*
  - split: test
    path: lr_bsz_scaling_law/test-*
- config_name: lr_bsz_scaling_law_modified
  data_files:
  - split: train
    path: lr_bsz_scaling_law_modified/train-*
  - split: test
    path: lr_bsz_scaling_law_modified/test-*
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
- config_name: sft_scaling_law
  data_files:
  - split: train
    path: sft_scaling_law/train-*
  - split: test
    path: sft_scaling_law/test-*
- config_name: vocab_scaling_law
  data_files:
  - split: train
    path: vocab_scaling_law/train-*
  - split: test
    path: vocab_scaling_law/test-*
---
