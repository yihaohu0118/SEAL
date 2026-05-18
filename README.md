# SEAL BFCL Release

This repository is a release-focused version of the AgentEvolver codebase for
the SEAL BFCL experiments. It keeps the components needed to reproduce the
paper's BFCL V3 multi-turn training and ablation results:

- verifier-grounded BFCL diagnostics,
- train-only schema affordance annotations,
- diagnosis-guided A-Patch advantage reweighting,
- strict BFCL validation logging and per-category analysis.

The release intentionally removes unrelated environments, game agents, and
experimental modules that are not used by `exp/bfcl_seal_04_full_seal.yaml`.

## Core Configs

The main release configs are under `exp/`:

| Config | Purpose |
| --- | --- |
| `bfcl_seal_00_vanilla_rl.yaml` | Vanilla GRPO baseline with sparse BFCL reward. |
| `bfcl_seal_01_wo_environment_side_adaptation.yaml` | A-Patch with BFCL diagnostics, without schema observation annotations. |
| `bfcl_seal_02_wo_diagnosis_guided_reweighting.yaml` | Observation annotations and diagnostics, without A-Patch weighting. |
| `bfcl_seal_03_wo_closed_loop_update.yaml` | Static interface setting used for the closed-loop ablation row. |
| `bfcl_seal_04_full_seal.yaml` | Full SEAL setting used for the main result. |

`bfcl_seal_04_full_seal.yaml` enables:

- `env_service.bfcl.observation_lite.enable=true`
- `task_manager.grader.original_grader=bfcl-dense-env`
- `tocf.feedback.dense_reward.enable=true`
- `tocf.advantage.apatch.enable=true`

## Launch

Start the BFCL environment service:

```bash
python launcher.py --with-bfcl
```

Run training with one of the release configs:

```bash
python launcher.py --conf exp/bfcl_seal_04_full_seal.yaml
```

Validation generations are written to:

```text
experiments/tech_synthetic/<experiment_name>/validation_log/
```

## Analyze BFCL Validation

Use the bundled script to compute per-category BFCL V3 multi-turn results:

```bash
BFCL_JSONL=env_service/environments/bfcl/bfcl_data/multi_turn_processed.jsonl

python scripts/stats_validation_bfcl.py \
  --val-dir experiments/tech_synthetic/bfcl_seal_04_full_seal/validation_log \
  --parquet data/bfcl_eval_400.parquet \
  --bfcl-jsonl "${BFCL_JSONL}"
```

The expected validation split contains 400 examples:

- `multi_turn_base`: 100
- `multi_turn_miss_func`: 100
- `multi_turn_miss_param`: 100
- `multi_turn_long_context`: 100

## Important Paths

```text
env_service/environments/bfcl/                 BFCL executable environment
env_service/environments/bfcl/multi_turn_progress.py
                                                verifier-grounded failure tags
agentevolver/module/task_manager/rewards/bfcl_dense_env_grader.py
                                                dense diagnostic metadata
agentevolver/module/tocf/apatch.py              diagnosis-guided weighting
agentevolver/module/tocf/state.py               persistent diagnostic state
agentevolver/module/tocf/stats.py               diagnostic statistics
scripts/stats_validation_bfcl.py                validation summarization
```

## Scope

This release is BFCL-only. It removes non-BFCL environments, game-arena code,
research prototypes, and unused TOCF patch families, leaving the components
needed to reproduce the SEAL ablations.
