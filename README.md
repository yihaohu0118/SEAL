<h1 align="center">
  SEAL: Synergistic Co-Evolution of Agents and Learning Environments
</h1>

<div align="center">

[![Homepage](https://img.shields.io/badge/Homepage-SEAL-blue.svg)](https://yihaohu0118.github.io/SEAL/)
[![Paper](https://img.shields.io/badge/Paper-PDF-red.svg)](https://yihaohu0118.github.io/SEAL/static/pdfs/seal-paper.pdf)
[![Poster](https://img.shields.io/badge/Poster-PDF-orange.svg)](https://yihaohu0118.github.io/SEAL/static/pdfs/seal-poster.pdf)
[![License](https://img.shields.io/badge/License-Apache--2.0-green.svg)](LICENSE)

</div>

<p align="center">
  <b>Tool-Use Agents · Self-Evolution · Reinforcement Learning</b>
</p>

<p align="center">
  Yihao Hu<sup>*,1,2</sup>, Zhihao Wen<sup>*,1</sup>, Xiujin Liu<sup>3</sup>, Pan Wang<sup>1,4</sup>, Xin Zhang<sup>1</sup>, Wei Wu<sup>1</sup>
</p>

<p align="center">
  <sup>*</sup>Equal Contribution · <sup>1</sup>Ant Group · <sup>2</sup>Westlake University · <sup>3</sup>University of Michigan-Ann Arbor · <sup>4</sup>University of Science and Technology of China
</p>

<p align="center">
  <img src="https://yihaohu0118.github.io/SEAL/static/images/seal-overview.png" alt="SEAL overview" width="86%">
</p>

## Overview

SEAL is a closed-loop co-evolution framework for interactive tool-use agents. It collects on-policy trajectories under executable verification, diagnoses failed rollouts into turn-level failure labels, and uses these diagnoses as a shared signal for both training-time interface evolution and model-side policy optimization.

In SEAL, the agent reveals its capability gaps, the learning interface adapts around these failures, and the policy internalizes the resulting feedback through GRPO. Evaluation remains strict: tool semantics, task labels, and the verifier are unchanged.

## Highlights

- **Verifier-grounded diagnosis:** executable traces are mapped to failure types such as invalid tool calls, argument mismatches, missing tool calls, recovery failures, and response mismatches.
- **Training-time interface evolution:** BFCL observations expose schema affordances and recovery-oriented feedback without changing the test-time environment.
- **Diagnosis-guided optimization:** diagnostic profiles reweight GRPO advantages while preserving the original verifier reward.

## Quick Start

```bash
git clone git@github.com:yihaohu0118/SEAL.git
cd SEAL

conda create -n seal python=3.10 -y
conda activate seal
pip install -r requirements.txt
```

Prepare and launch the BFCL environment:

```bash
bash env_service/environments/bfcl/setup.sh
conda activate bfcl
bash env_service/launch_script/bfcl.sh
```

Run the SEAL training recipe:

```bash
conda activate seal
python launcher.py --conf exp/SEAL.yaml
```

## Repository Layout

```text
exp/SEAL.yaml                         Full SEAL training configuration
env_service/environments/bfcl/        BFCL executable environment and interface adaptation
agentevolver/module/tocf/             Diagnostic state and advantage reweighting
agentevolver/module/task_manager/     Rewards, task adapters, and BFCL grader
data/                                 Released BFCL train/evaluation split files
```

## Citation

```bibtex
@article{hu2026seal,
  title={SEAL: Synergistic Co-Evolution of Agents and Learning Environments},
  author={Hu, Yihao and Wen, Zhihao and Liu, Xiujin and Wang, Pan and Zhang, Xin and Wu, Wei},
  journal={Preprint},
  year={2026},
  url={https://yihaohu0118.github.io/SEAL/}
}
```

## License

This project is released under the [Apache License 2.0](LICENSE).
