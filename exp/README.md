# BFCL SEAL Ablation Experiments

This folder contains the paper-facing BFCL V3 multi-turn ablation configs.
The base-model-only row is intentionally omitted.

Each YAML is expanded as a standalone experiment entry and does not inherit
from `examples/`. The only defaults are the repository-level trainer/runtime
defaults: `ppo_trainer` and `agentevolver`.

| Paper setting | Config |
| --- | --- |
| + Vanilla RL | `bfcl_seal_00_vanilla_rl.yaml` |
| w/o Environment-Side Adaptation | `bfcl_seal_01_wo_environment_side_adaptation.yaml` |
| w/o Diagnosis-Guided Reweighting | `bfcl_seal_02_wo_diagnosis_guided_reweighting.yaml` |
| w/o Closed-Loop Update | `bfcl_seal_03_wo_closed_loop_update.yaml` |
| Full SEAL | `bfcl_seal_04_full_seal.yaml` |

Example:

```bash
python -m agentevolver.main_ppo \
  --config-path exp \
  --config-name bfcl_seal_04_full_seal
```
