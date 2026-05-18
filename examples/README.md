# Minimal BFCL Example Configs

The release keeps only a small set of reusable BFCL configs. The paper
ablation configs live in `exp/` and are the recommended entry points.

| Config | Role |
| --- | --- |
| `bfcl_grpo_base.yaml` | Sparse GRPO BFCL baseline. |
| `bfcl_tocf_taes_base.yaml` | Diagnostic-control entry point. |
| `bfcl_tocf_taes_shared.yaml` | Shared diagnostic/A-Patch base for compact examples. |
| `bfcl_grpo_apatch.yaml` | A-Patch only example. |
| `bfcl_grpo_apatch_observation_required_rerun.yaml` | A-Patch + train-only required/enum schema annotations. |

For release reproduction, prefer:

```bash
python launcher.py --conf exp/bfcl_seal_04_full_seal.yaml
```
