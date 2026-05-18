from __future__ import annotations

import json
import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Iterable

from agentevolver.module.tocf.category import infer_task_category
from agentevolver.module.tocf.state import TOCFCapabilityState
from agentevolver.schema.task import Task
from agentevolver.schema.trajectory import Trajectory


@dataclass
class CategoryStats:
    count: int = 0
    success: int = 0
    reward_sum: float = 0.0
    partial_count: int = 0

    def observe(self, reward: float, success: bool):
        self.count += 1
        self.success += int(success)
        self.reward_sum += reward
        self.partial_count += int(0.0 < reward < 1.0)

    def snapshot(self) -> dict[str, float | int]:
        return {
            "count": self.count,
            "success": self.success,
            "success_rate": self.success / self.count if self.count else 0.0,
            "reward_mean": self.reward_sum / self.count if self.count else 0.0,
            "partial_count": self.partial_count,
        }


class TOCFStats:
    """Pure statistics collector for TOCF. It does not mutate training behavior."""

    def __init__(
        self,
        dump_dir: str | None = None,
        capability_state: TOCFCapabilityState | None = None,
    ):
        self.dump_dir = dump_dir
        self.capability_state = capability_state or TOCFCapabilityState()
        self.total = defaultdict(CategoryStats)
        self.window = defaultdict(CategoryStats)
        self.error_patterns = Counter()
        self.last_epoch: int | str | None = None
        self.last_step: int | None = None

    def _task_for_trajectory(self, tasks: list[Task], trajectory: Trajectory) -> Task | None:
        try:
            idx = int(trajectory.data_id)
            return tasks[idx]
        except Exception:
            return None

    def _reward_tuple(self, trajectory: Trajectory) -> tuple[float, bool]:
        if trajectory.reward is None:
            return 0.0, False
        reward = float(trajectory.reward.outcome)
        success = bool(trajectory.reward.success_rate >= 1.0 or reward >= 1.0)
        return reward, success

    def _observe_errors(self, trajectory: Trajectory):
        for step in getattr(trajectory, "steps", []) or []:
            content = step.get("content", "") if isinstance(step, dict) else ""
            if not content:
                continue
            for pattern in (r"\[ERROR\][^\n]*", r'"error"\s*:\s*"([^"]+)"', r"Error from environment:[^\]]+"):
                for match in re.findall(pattern, content):
                    value = match if isinstance(match, str) else str(match)
                    self.error_patterns[value[:200]] += 1

    def _progress_info(self, trajectory: Trajectory) -> dict[str, Any]:
        reward_meta = (
            getattr(getattr(trajectory, "reward", None), "metadata", None) or {}
        )
        return reward_meta.get("bfcl_dense_progress_info", {}) or {}

    def observe(self, tasks: Iterable[Task], trajectories: Iterable[Trajectory], epoch: int | str, global_step: int):
        task_list = list(tasks)
        self.last_epoch = epoch
        self.last_step = global_step
        for trajectory in trajectories:
            task = self._task_for_trajectory(task_list, trajectory)
            if task is None:
                continue
            category = infer_task_category(task.task_id, task.env_type, task.metadata)
            reward, success = self._reward_tuple(trajectory)
            self.total[category].observe(reward, success)
            self.window[category].observe(reward, success)
            progress_info = self._progress_info(trajectory)
            failure_tags = list(progress_info.get("failure_tags") or [])
            self.capability_state.observe(
                category=category,
                task_id=task.task_id,
                reward=reward,
                success=success,
                failure_tags=failure_tags,
                epoch=epoch,
                global_step=global_step,
            )
            self._observe_errors(trajectory)

    def snapshot(self, *, window: bool = True) -> dict[str, Any]:
        source = self.window if window else self.total
        capability = self.capability_state.snapshot(window=window)
        return {
            "epoch": self.last_epoch,
            "global_step": self.last_step,
            "categories": {category: stats.snapshot() for category, stats in source.items()},
            "tags": capability.get("tags", {}),
            "category_tags": capability.get("category_tags", {}),
            "turn_positions": capability.get("turn_positions", {}),
            "tasks": capability.get("tasks", {}),
            "dynamic_tag_weights": capability.get("dynamic_tag_weights", {}),
            "top_errors": self.error_patterns.most_common(20),
        }

    def metrics(self, prefix: str = "tocf/window") -> dict[str, float]:
        metrics = {}
        for category, stats in self.window.items():
            snap = stats.snapshot()
            base = f"{prefix}/{category}"
            metrics[f"{base}/count"] = float(snap["count"])
            metrics[f"{base}/success_rate"] = float(snap["success_rate"])
            metrics[f"{base}/reward_mean"] = float(snap["reward_mean"])
            metrics[f"{base}/partial_count"] = float(snap["partial_count"])
        metrics.update(self.capability_state.metrics())
        return metrics

    def reset_window(self):
        self.window = defaultdict(CategoryStats)
        self.error_patterns.clear()
        self.capability_state.reset_window()

    def dump(self, name: str):
        if not self.dump_dir:
            return
        os.makedirs(self.dump_dir, exist_ok=True)
        path = os.path.join(self.dump_dir, name)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.snapshot(window=True), f, ensure_ascii=False, indent=2)
