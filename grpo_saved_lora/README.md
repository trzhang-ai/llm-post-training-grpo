---
base_model: unsloth/qwen2.5-3b-instruct-unsloth-bnb-4bit
library_name: peft
pipeline_tag: text-generation
tags:
- base_model:adapter:unsloth/qwen2.5-3b-instruct-unsloth-bnb-4bit
- grpo
- lora
- transformers
- trl
- unsloth
---

# Qwen2.5 Letter-Counting GRPO LoRA Adapter

## Summary

This PEFT adapter records a bounded post-training experiment on structured
letter counting. It applies LoRA to a 4-bit Qwen2.5-3B-Instruct base model and
uses GRPO with deterministic, programmatic rewards for response structure and
counting correctness.

This artifact is an experiment output, not a general-purpose upgraded Qwen
model and not a production deployment.

## Model details

| Field | Value |
| --- | --- |
| Base model | `unsloth/qwen2.5-3b-instruct-unsloth-bnb-4bit` |
| Adapter type | PEFT LoRA |
| Rank / alpha | 16 / 16 |
| Target modules | `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` |
| Trainable parameters | 29,933,568 |
| Observed total parameters | 3,115,872,256 |
| Trainable share | 0.96% |
| Adapter format | safetensors |
| Framework snapshot | PEFT 0.17.1, TRL, Transformers, Unsloth, vLLM |

## Intended use

The adapter is intended for technical review of a small, rule-verifiable GRPO
experiment and for reproducing the saved letter-counting comparisons in the
associated notebook. It may help illustrate LoRA configuration, programmatic
reward design, training monitoring, and PEFT artifact export.

It should not be used as evidence of general reasoning improvement, held-out
task accuracy, safety, catastrophic-forgetting resistance, or production model
quality.

## Training data and objective

Training used 401 generated records, each pairing a word with a target letter.
The target response spells the word one character per numbered line, maintains
a running occurrence count, and ends with a final count. Five deterministic
reward functions score numbering, spelling, running counts, output format, and
final-answer correctness.

The longer run used:

- 200 training steps;
- learning rate `0.0001`;
- beta `0.001`;
- per-device batch size `4`;
- four generations per prompt;
- gradient accumulation steps `1`;
- one observed NVIDIA Tesla T4;
- saved runtime of 29 minutes 12 seconds.

## Evaluation

The retained training table shows mean total reward increasing from `3.86881`
over the first 20 steps to `4.77946` over the last 20. Mean final-answer reward
increased from `0.7125` to `0.95` over the same windows.

These are training-batch metrics, not held-out task accuracy. The only saved
base-versus-adapter counting comparison was answered correctly by both models,
so it does not establish that the adapter outperformed the base model. One
capital-city prompt is only a smoke test.

## Limitations

- No held-out evaluation, multi-seed comparison, ablation, or confidence
  interval is available.
- The saved Chain-of-Thought baseline example still miscounted `engage`.
- The training data are generated and narrow in scope.
- Programmatic rewards are not human preferences or a learned reward model;
  this is not RLHF.
- The repository records observed package versions but does not provide a
  tested, fully locked GPU environment.
- Base-model risks, license terms, and limitations continue to apply.

## Reproduction record

See `../gen_ai_fundamentals_project_starter.ipynb` for the retained setup,
reward checks, saved training outputs, plots, and comparisons. See
`../PROJECT_EVIDENCE.md` for exact artifact hashes and claim boundaries.

## Provenance and rights

Tianrui Zhang produced this adapter while completing a scaffolded Udacity
learning project. No new license is granted for the adapter, base model,
starter materials, or third-party code; consult each upstream source's terms
before reuse or redistribution.
