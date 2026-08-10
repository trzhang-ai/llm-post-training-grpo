# Project evidence

Reviewed on 2026-08-10 against the archived notebook, saved adapter, signed-in
Udacity instructions, and project rubric.

## Artifact integrity

| Artifact | SHA-256 |
|---|---|
| `gen_ai_fundamentals_project_starter.ipynb` | `8b04d3c2fbeb32409461b242966965a2422fba7dcd945f263c9277302fa39162` |
| `grpo_saved_lora/README.md` | `a8e1eaee63a10c975bf72d940143942c6ab10002b1c2bddaafe2f8a9069ff60a` |
| `grpo_saved_lora/adapter_config.json` | `bd6eede7fc52a400219b06587b1d59bcfa6ae3043ffd45808879c8e6178811a9` |
| `grpo_saved_lora/adapter_model.safetensors` | `fcf0a5dbe652af529643d739b6ca5a9083ad68da22459b19d74609c95de38408` |

## Rubric mapping

| Rubric area | Archived evidence | Boundary |
|---|---|---|
| LoRA model setup | Rank 16; seven projection modules; model instantiated and trained | Adapter training, not full-model training |
| Prompting baseline | Blank and one-example Chain-of-Thought runs saved | The saved CoT example remained incorrect |
| Reward design | Numbering, spelling, counting, format, and final-correctness functions | Deterministic rewards, not a learned reward model |
| Reward validation | Better paired sample scored higher for all five functions | Unit-like examples, not task accuracy |
| Longer training | 200 saved steps, approximately 29 minutes, reward table and plot | Training batches, not held-out evaluation |
| Final comparison | Base and adapter outputs saved for one counting prompt | Both were correct; no superiority claim |
| Knowledge retention | Both answered Paris for the capital of France | One-question smoke test only |

## Reward validation outputs

| Reward | Lower-quality sample | Higher-quality sample |
|---|---:|---:|
| Numbering | -0.3333 | 0.5 |
| Spelling | 0.8 | 1.0 |
| Running count | -0.6 | 1.0 |
| Format | 0.0 | 1.0 |
| Final answer | 0.0 | 1.0 |

## Longer-run configuration

- dataset records: 401
- steps: 200
- learning rate: 0.0001
- beta: 0.001
- per-device batch size: 4
- generations per prompt: 4
- gradient accumulation steps: 1
- saved runtime: 29 minutes 12 seconds

## Longer-run metrics

| Training-batch metric | Step 1 | Step 200 | First-20 mean | Last-20 mean |
|---|---:|---:|---:|---:|
| Total reward | 3.223145 | 4.558044 | 3.86881 | 4.77946 |
| Final-answer reward | 0.25 | 1.0 | 0.7125 | 0.95 |

## Safe project description

> Completed a scaffolded Udacity LLM post-training project using 4-bit LoRA
> and GRPO to adapt Qwen2.5-3B-Instruct for letter counting; implemented five
> programmatic reward functions, ran a 200-step training session over 401
> generated examples on a Tesla T4, monitored reward trends, and saved a PEFT
> adapter. Results are training-batch evidence, not held-out accuracy or
> production deployment.

## Unsupported claims

- production fine-tuning or production deployment
- RLHF or human-preference-data work
- learned reward-model training
- base-model pretraining or full-parameter fine-tuning
- three billion parameters trained
- proven accuracy improvement or held-out generalization
- no catastrophic forgetting
- from-scratch model-training architecture
- novel reinforcement-learning research
- Udacity employment or sponsorship
