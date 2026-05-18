from __future__ import annotations

import re
from typing import Any

from agentevolver.client.env_client import EnvClient
from agentevolver.module.agent_flow.reward_calculator import GraderResult, RewardCalculator
from agentevolver.schema.task import Task
from agentevolver.schema.trajectory import Trajectory

from . import grader_manager


def _cfg_get(config: Any, path: str, default=None):
    cur = config
    for part in path.split("."):
        if cur is None:
            return default
        if isinstance(cur, dict):
            cur = cur.get(part, default)
        else:
            cur = getattr(cur, part, default)
    return cur


@grader_manager.reg("bfcl-dense-env")
class BfclDenseEnvGrader(RewardCalculator):
    """BFCL environment grader with outcome-aligned per-turn progress.

    Two modes are supported via ``tocf.feedback.dense_reward.mode``:

    * ``capped`` (default, legacy): ``score = min(cap, weight * progress)``
      when the trajectory does not fully pass, ``1.0`` otherwise.
      The cap defaults to ``0.7``.

    * ``t3rl``: per-turn progress reward inspired by
      ``/Users/huyihao/Desktop/T3RL-main/t3rl/rollout/bfcl.py:res_to_sample``
      but hardened against a well-known failure mode on small training
      sets (see *abstention exploit* below).

      Default formula::

          score = (passed_scorable + w * passed_irrelevance)
                / (scorable       + w * irrelevance      )

      where ``w = irrelevance_weight`` (default ``0.5``). Setting
      ``w = 1.0`` reproduces the raw T3RL ``progress_with_irrelevance``
      signal (line-for-line with
      ``multi_turn_progress.compute_per_turn_progress``).

      Why not use T3RL's ``w = 1.0`` directly?  BFCL multi_turn_miss_func
      tasks mix GT-empty "abstention" turns with GT-nonempty "reveal"
      turns. Under ``w = 1.0`` a trajectory that abstains on *every*
      turn earns a non-trivial reward (e.g. ``2/5 = 0.40`` for a 2-irr /
      3-scorable task), which is strictly higher than an honest-but-
      failing trajectory that scores ``0/3 = 0.0`` on the scorable
      half. Empirically, GRPO on Qwen2.5-7B + a 400-sample BFCL split
      locks onto that abstention attractor around step ~100 and the
      policy collapses across *all* categories, not just miss_func
      (observed: 9% → 14.5% → 8.75% over 150 steps). T3RL itself
      reports convergence only with a much larger corpus and many more
      update steps, so we borrow the shaping idea but rescale the
      irrelevance contribution so partial-success rollouts always
      dominate pure abstention in the GRPO advantage.

      Optional ``scorable_floor`` (default ``False``) is a harder
      version: zero out the dense reward whenever ``passed_scorable ==
      0`` and ``scorable_turns > 0``. Useful in late training once the
      model can already clear at least one scorable turn most of the
      time. Leave it off early (all-zero rewards in a group kill the
      gradient).

    Historical note
    ---------------
    The first version of this grader clamped every non-passing trajectory
    to ``0.0`` because BFCL's multi_turn accuracy is binary, producing a
    "dense" grader that was in fact sparse. The second version shaped the
    reward with *process* statistics (turn_coverage_rate, tool_call_accept_rate,
    tool_exec_success_rate). Those signals are not monotone-aligned with
    correctness: a model can score high on them by emitting any well-formed
    tool call, which actively hurts ``miss_func`` tasks where the correct
    behaviour is to ask rather than call. The current capped/t3rl modes
    use the *same* checker semantics as the final binary accuracy, just
    evaluated per user turn.
    """

    def __init__(self, task: Task):
        super().__init__(task)
        self._config = None

    def set_config(self, config):
        self._config = config

    # ---------------------------------------------------------------- utils
    def _training_progress(self) -> float:
        epoch_value = (self.task.metadata or {}).get("epoch")
        if epoch_value is None:
            return 0.0

        epoch_index = None
        if isinstance(epoch_value, int):
            epoch_index = epoch_value
        elif isinstance(epoch_value, str):
            match = re.match(r"train\.(\d+)\.", epoch_value)
            if match:
                epoch_index = int(match.group(1))

        if epoch_index is None:
            return 0.0

        total_epochs = float(_cfg_get(self._config, "trainer.total_epochs", 1) or 1)
        return min(1.0, max(0.0, epoch_index / max(1.0, total_epochs)))

    def _partial_credit_cap(self) -> float:
        cfg_base = "tocf.feedback.dense_reward"
        cap = float(_cfg_get(self._config, f"{cfg_base}.partial_credit_cap", 0.7))
        if bool(_cfg_get(self._config, f"{cfg_base}.cap_decay", False)):
            min_cap = float(_cfg_get(self._config, f"{cfg_base}.min_cap", 0.1))
            cap = max(min_cap, cap * (1.0 - self._training_progress()))

        metadata_cfg = ((self.task.metadata or {}).get("tocf", {}) or {}).get("dense_reward", {})
        if "partial_credit_cap" in metadata_cfg:
            cap = float(metadata_cfg["partial_credit_cap"])
        return max(0.0, min(1.0, cap))

    def _partial_credit_weight(self) -> float:
        cfg_base = "tocf.feedback.dense_reward"
        weight = float(_cfg_get(self._config, f"{cfg_base}.partial_credit_weight", 1.0))
        metadata_cfg = ((self.task.metadata or {}).get("tocf", {}) or {}).get("dense_reward", {})
        if "partial_credit_weight" in metadata_cfg:
            weight = float(metadata_cfg["partial_credit_weight"])
        return max(0.0, weight)

    def _mode(self) -> str:
        """Reward-shaping mode. ``capped`` keeps the legacy min(cap, w*p)
        formula; ``t3rl`` mirrors T3RL-main's uncapped progress-with-
        irrelevance signal. Task metadata can override the config so
        individual tasks can opt into a different mode if needed.
        """
        cfg_base = "tocf.feedback.dense_reward"
        mode = str(_cfg_get(self._config, f"{cfg_base}.mode", "capped") or "capped").lower()
        metadata_cfg = ((self.task.metadata or {}).get("tocf", {}) or {}).get("dense_reward", {})
        if "mode" in metadata_cfg:
            mode = str(metadata_cfg["mode"] or "capped").lower()
        if mode not in ("capped", "t3rl"):
            mode = "capped"
        return mode

    def _irrelevance_weight(self) -> float:
        """Multiplier on the irrelevance-turn numerator AND denominator
        in the ``t3rl`` formula.  ``1.0`` matches raw T3RL; ``0.5``
        (default) gives correctly-abstained turns only half the weight
        of scorable turns, which empirically removes the abstention-only
        GRPO attractor on small BFCL splits.
        """
        cfg_base = "tocf.feedback.dense_reward"
        w = float(_cfg_get(self._config, f"{cfg_base}.irrelevance_weight", 0.5) or 0.5)
        metadata_cfg = ((self.task.metadata or {}).get("tocf", {}) or {}).get("dense_reward", {})
        if "irrelevance_weight" in metadata_cfg:
            w = float(metadata_cfg["irrelevance_weight"])
        return max(0.0, w)

    def _scorable_floor(self) -> bool:
        """If True, zero the dense reward whenever no scorable turn
        passes (``passed_turns == 0`` and ``scorable_turns > 0``).  The
        strict version of the abstention guard; off by default to avoid
        all-zero GRPO groups during very early training.
        """
        cfg_base = "tocf.feedback.dense_reward"
        flag = bool(_cfg_get(self._config, f"{cfg_base}.scorable_floor", False))
        metadata_cfg = ((self.task.metadata or {}).get("tocf", {}) or {}).get("dense_reward", {})
        if "scorable_floor" in metadata_cfg:
            flag = bool(metadata_cfg["scorable_floor"])
        return flag

    # ------------------------------------------------------------- scoring
    def calculate_reward(
        self, trajectory: Trajectory, env: EnvClient, instance_id: str
    ) -> GraderResult:
        result = env.evaluate(instance_id, params={"sparse": False})
        if isinstance(result, dict):
            raw_accuracy = float(result.get("accuracy", 0.0) or 0.0)
            completed = bool(result.get("completed", False))
        else:
            raw_accuracy = float(result or 0.0)
            completed = raw_accuracy >= 1.0
            result = {"accuracy": raw_accuracy, "completed": completed}

        # Outcome-aligned partial credit: fraction of GT user turns passed.
        # Populated by env_handler.evaluate() for multi_turn categories;
        # missing (single_turn / relevance) defaults to 0.0 and the branch
        # below falls back to the binary accuracy.
        progress = float(result.get("bfcl_progress", 0.0) or 0.0)
        progress = max(0.0, min(1.0, progress))
        # T3RL-style metric that also scores irrelevance turns. Falls back
        # to ``progress`` when env_handler did not emit it (e.g. older
        # cache entries or single-turn categories).
        progress_t3rl = float(
            result.get("bfcl_progress_with_irrelevance", progress) or 0.0
        )
        progress_t3rl = max(0.0, min(1.0, progress_t3rl))
        progress_info = result.get("bfcl_progress_info", {}) or {}

        mode = self._mode()

        if mode == "t3rl":
            # Hardened T3RL progress. See class docstring for the
            # derivation; TL;DR: naive ``progress_with_irrelevance``
            # lets GRPO learn an abstain-everywhere local optimum, so
            # we scale the irrelevance contribution by ``w < 1``.
            passed_sc = int(progress_info.get("passed_turns", 0) or 0)
            sc = int(progress_info.get("scorable_turns", 0) or 0)
            passed_ir = int(progress_info.get("passed_irrelevance_turns", 0) or 0)
            ir = int(progress_info.get("irrelevance_turns", 0) or 0)
            w = self._irrelevance_weight()
            scorable_floor_triggered = (
                self._scorable_floor() and sc > 0 and passed_sc == 0
            )

            if raw_accuracy >= 1.0:
                # Full-pass safety net: BFCL's binary accuracy is
                # authoritative, so a fully-correct trajectory always
                # scores 1.0 even if the per-turn scorer disagreed.
                score = 1.0
            elif scorable_floor_triggered:
                score = 0.0
            else:
                num = float(passed_sc) + w * float(passed_ir)
                den = float(sc) + w * float(ir)
                if den > 0.0:
                    score = num / den
                else:
                    # No GT turns at all: fall back to the raw
                    # progress_with_irrelevance (unreachable for
                    # multi_turn categories, defensive for single-turn).
                    score = progress_t3rl
                score = max(0.0, min(1.0, score))
            cap = 1.0
            weight = w
        else:
            # Legacy capped mode.
            cap = self._partial_credit_cap()
            weight = self._partial_credit_weight()
            if raw_accuracy >= 1.0:
                score = 1.0
            else:
                score = min(cap, weight * progress)

        metadata = {
            "bfcl_dense_raw": {
                k: v for k, v in result.items() if k != "bfcl_progress_info"
            },
            "bfcl_dense_raw_accuracy": raw_accuracy,
            "bfcl_dense_cap": cap,
            "bfcl_dense_weight": weight,  # t3rl: irrelevance_weight; capped: partial_credit_weight
            "bfcl_dense_completed": completed,
            "bfcl_dense_progress": progress,
            "bfcl_dense_progress_t3rl": progress_t3rl,
            "bfcl_dense_mode": mode,
            "bfcl_dense_progress_info": progress_info,
        }
        if mode == "t3rl":
            reason = (
                f"BFCL dense[t3rl]: score={score:.4f}, "
                f"raw_accuracy={raw_accuracy:.4f}, "
                f"progress_with_irrelevance={progress_t3rl:.4f}, "
                f"legacy_progress={progress:.4f}, completed={completed}, "
                f"passed/scorable="
                f"{progress_info.get('passed_turns', '-')}/"
                f"{progress_info.get('scorable_turns', '-')}, "
                f"irrelevance(passed/total)="
                f"{progress_info.get('passed_irrelevance_turns', '-')}/"
                f"{progress_info.get('irrelevance_turns', '-')}, "
                f"irr_weight={weight:.2f}"
            )
        else:
            reason = (
                f"BFCL dense[capped]: score={score:.4f}, "
                f"raw_accuracy={raw_accuracy:.4f}, progress={progress:.4f}, "
                f"cap={cap:.4f}, completed={completed}, "
                f"passed/scorable="
                f"{progress_info.get('passed_turns', '-')}/"
                f"{progress_info.get('scorable_turns', '-')}"
            )
        return {"score": score, "reason": reason, "metadata": metadata}
