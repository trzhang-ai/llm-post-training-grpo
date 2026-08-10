# Qwen2.5 GRPO Letter Counting

Private archival repository for the Udacity project **Teaching an LLM to
Count!** from **Generative AI Fundamentals**.

## Status

- Project type: completed Udacity learning project using starter scaffolding
- Classroom result: `Passed`, reverified on 2026-08-10
- Publication status: private evidence archive
- Production status: not deployed and not production evidence

## Project goal

The project adapts Qwen2.5-3B-Instruct to count occurrences of a target letter
by following a letter-by-letter procedure. It uses parameter-efficient
fine-tuning and programmatic rewards rather than updating the full base model.

## Repository contents

- `gen_ai_fundamentals_project_starter.ipynb` — completed notebook with saved
  setup, prompting, reward validation, training, plots, and comparison outputs
- `grpo_saved_lora/adapter_config.json` — PEFT/LoRA configuration
- `grpo_saved_lora/adapter_model.safetensors` — saved LoRA adapter, stored with
  Git LFS
- `grpo_saved_lora/README.md` — original generated adapter model-card template
- `PROJECT_EVIDENCE.md` — rubric-aligned evidence and limitations
- `CHECKSUMS.sha256` — source-artifact integrity hashes

The original source directory was named `grpo_saved_lora.zip` even though it
was a directory. This repository uses `grpo_saved_lora/`, matching the path
referenced by the notebook, while preserving the files byte-for-byte.

## Observed implementation

- Qwen2.5-3B-Instruct loaded in 4-bit mode through Unsloth
- LoRA rank 16 and alpha 16
- LoRA applied to `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`,
  `up_proj`, and `down_proj`
- 29,933,568 trainable parameters out of 3,115,872,256 observed parameters
  (0.96 percent)
- 401 synthetic word-and-letter counting records using Hugging Face Datasets
- five deterministic rewards covering numbering, spelling, running counts,
  response format, and final-answer correctness
- GRPO training through Hugging Face TRL with vLLM-accelerated generation
- 30-step quick run and 200-step longer run on one NVIDIA Tesla T4
- LoRA adapter saved as safetensors and reloaded through PEFT

## Saved training evidence

The 200-step run completed in approximately 29 minutes. Metrics below are
training-batch signals parsed from the saved notebook table, not held-out
accuracy.

| Metric | First 20 steps | Last 20 steps |
|---|---:|---:|
| Mean total reward | 3.86881 | 4.77946 |
| Mean final-answer reward | 0.7125 | 0.95 |

## Evidence boundaries

- The saved Chain-of-Thought baseline followed the requested structure but
  still returned the wrong count for `engage`.
- The final comparison used one generated dataset item; both the base and
  adapted models answered it correctly, so it does not establish adapter
  superiority or generalization.
- One capital-city question was used as a knowledge-retention smoke test. It
  does not prove the absence of catastrophic forgetting.
- No held-out accuracy, multi-seed evaluation, ablation, or statistical
  uncertainty is saved.
- Rewards are deterministic Python functions, not human preference data or a
  learned reward model. This project should not be described as RLHF.
- This is not base-model pretraining, full-parameter fine-tuning, production
  deployment, or novel reinforcement-learning research.

## Observed runtime

The saved notebook output reported:

- NVIDIA Tesla T4 with 14.563 GiB observed VRAM
- Python 3.12 environment supplied by the Udacity/Vocareum workspace
- Unsloth 2025.9.7
- Transformers 4.55.4
- vLLM 0.10.1
- PyTorch 2.7.1 with CUDA 12.6

These are observations, not a pinned reproducible environment. The repository
does not include a lockfile or complete dependency manifest.

## Source pages

- [Project overview](https://learn.udacity.com/nd608?version=3.2.31&partKey=cd13303&lessonKey=998b15fd-00fc-4eac-9741-ab2f306a5653&conceptKey=8be2f263-f1d8-484b-ba3a-9fa9059fcc40)
- [Instructions](https://learn.udacity.com/nd608?version=3.2.31&partKey=cd13303&lessonKey=998b15fd-00fc-4eac-9741-ab2f306a5653&conceptKey=2dc10354-5b91-48f1-9256-6e3faf586c99)
- [Project rubric](https://learn.udacity.com/nd608?version=3.2.31&partKey=cd13303&lessonKey=998b15fd-00fc-4eac-9741-ab2f306a5653&project=rubric)

## Rights and reuse

This repository is an owner-only archive of a learning project. No new license
is granted for Udacity starter materials, model weights, or third-party code;
their original terms continue to apply.
