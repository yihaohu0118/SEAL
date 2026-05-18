from __future__ import annotations

import json
import os
from collections import Counter
from dataclasses import dataclass
from typing import Any


EXCLUDED_FAILURE_TAGS = frozenset({"checker_error", "gt_error"})
UNKNOWN_TAG = "unknown"
PASS_TAG = "pass"


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def dominant_failure_tag(tags: list[str] | None) -> str:
    if not tags:
        return UNKNOWN_TAG
    non_pass = [str(t) for t in tags if t and t != PASS_TAG]
    if non_pass:
        return Counter(non_pass).most_common(1)[0][0]
    if any(t == PASS_TAG for t in tags):
        return PASS_TAG
    return UNKNOWN_TAG


def turn_position_bucket(index: int, total: int) -> str:
    if total <= 1:
        return "only"
    ratio = float(index + 1) / float(total)
    if ratio <= 1.0 / 3.0:
        return "early"
    if ratio <= 2.0 / 3.0:
        return "middle"
    return "late"


def pack_category_tag(category: str, tag: str) -> str:
    return f"{category}::{tag}"


def unpack_category_tag(key: str) -> tuple[str, str]:
    if "::" not in key:
        return key, UNKNOWN_TAG
    category, tag = key.split("::", 1)
    return category, tag


@dataclass
class CapabilityStats:
    count: int = 0
    success: int = 0
    reward_sum: float = 0.0
    partial_count: int = 0

    def observe(self, reward: float, success: bool) -> None:
        self.count += 1
        self.success += int(success)
        self.reward_sum += float(reward)
        self.partial_count += int(0.0 < reward < 1.0)

    @property
    def success_rate(self) -> float:
        return float(self.success) / float(self.count) if self.count else 0.0

    @property
    def reward_mean(self) -> float:
        return float(self.reward_sum) / float(self.count) if self.count else 0.0

    def snapshot(self) -> dict[str, float | int]:
        return {
            "count": self.count,
            "success": self.success,
            "success_rate": self.success_rate,
            "reward_mean": self.reward_mean,
            "partial_count": self.partial_count,
        }

    def to_dict(self) -> dict[str, float | int]:
        return {
            "count": int(self.count),
            "success": int(self.success),
            "reward_sum": float(self.reward_sum),
            "partial_count": int(self.partial_count),
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any] | None) -> "CapabilityStats":
        raw = raw or {}
        return cls(
            count=int(raw.get("count", 0) or 0),
            success=int(raw.get("success", 0) or 0),
            reward_sum=_safe_float(raw.get("reward_sum", 0.0)),
            partial_count=int(raw.get("partial_count", 0) or 0),
        )


class TOCFCapabilityState:
    """Shared persistent capability state for TOCF patches."""

    def __init__(self, path: str | None = None):
        self.path = path
        self.total_categories: dict[str, CapabilityStats] = {}
        self.window_categories: dict[str, CapabilityStats] = {}
        self.total_tags: dict[str, CapabilityStats] = {}
        self.window_tags: dict[str, CapabilityStats] = {}
        self.total_category_tags: dict[str, CapabilityStats] = {}
        self.window_category_tags: dict[str, CapabilityStats] = {}
        self.total_turn_positions: dict[str, CapabilityStats] = {}
        self.window_turn_positions: dict[str, CapabilityStats] = {}
        self.tasks: dict[str, dict[str, Any]] = {}
        self.dynamic_tag_weights: dict[str, float] = {}
        self.last_epoch: int | str | None = None
        self.last_step: int | None = None
        self.observe_count = 0

    @classmethod
    def load(cls, path: str | None) -> "TOCFCapabilityState":
        state = cls(path=path)
        if not path or not os.path.exists(path):
            return state
        try:
            with open(path, "r", encoding="utf-8") as handle:
                raw = json.load(handle)
        except (OSError, json.JSONDecodeError):
            return state
        state._load_dict(raw)
        return state

    def _stats_bucket(self, bucket: dict[str, CapabilityStats], key: str) -> CapabilityStats:
        if key not in bucket:
            bucket[key] = CapabilityStats()
        return bucket[key]

    def observe(
        self,
        *,
        category: str,
        task_id: str | None,
        reward: float,
        success: bool,
        failure_tags: list[str] | None,
        epoch: int | str | None,
        global_step: int | None,
    ) -> str:
        tags = [str(tag) for tag in (failure_tags or []) if tag]
        if not tags:
            tags = [UNKNOWN_TAG]
        dominant_tag = dominant_failure_tag(tags)
        category = category or "_none"
        task_key = str(task_id or "")

        self.last_epoch = epoch
        self.last_step = global_step
        self.observe_count += 1

        self._stats_bucket(self.total_categories, category).observe(reward, success)
        self._stats_bucket(self.window_categories, category).observe(reward, success)

        for tag in tags:
            self._stats_bucket(self.total_tags, tag).observe(reward, success)
            self._stats_bucket(self.window_tags, tag).observe(reward, success)
            ct_key = pack_category_tag(category, tag)
            self._stats_bucket(self.total_category_tags, ct_key).observe(reward, success)
            self._stats_bucket(self.window_category_tags, ct_key).observe(reward, success)

        for index, tag in enumerate(tags):
            bucket = turn_position_bucket(index, len(tags))
            key = pack_category_tag(bucket, tag)
            self._stats_bucket(self.total_turn_positions, key).observe(reward, success)
            self._stats_bucket(self.window_turn_positions, key).observe(reward, success)

        if task_key:
            task_stats = self.tasks.setdefault(
                task_key,
                {
                    "task_id": task_key,
                    "category": category,
                    "count": 0,
                    "success": 0,
                    "reward_sum": 0.0,
                    "last_reward": 0.0,
                    "last_success": False,
                    "last_tag": UNKNOWN_TAG,
                    "tag_counts": {},
                    "last_step": None,
                },
            )
            task_stats["category"] = category
            task_stats["count"] = int(task_stats.get("count", 0) or 0) + 1
            task_stats["success"] = int(task_stats.get("success", 0) or 0) + int(success)
            task_stats["reward_sum"] = _safe_float(task_stats.get("reward_sum", 0.0)) + float(reward)
            task_stats["last_reward"] = float(reward)
            task_stats["last_success"] = bool(success)
            task_stats["last_tag"] = dominant_tag
            task_stats["last_step"] = global_step
            tag_counts = task_stats.setdefault("tag_counts", {})
            for tag in tags:
                tag_counts[tag] = int(tag_counts.get(tag, 0) or 0) + 1

        return dominant_tag

    def update_dynamic_tag_weights(
        self,
        base_weights: dict[str, float],
        *,
        min_scale: float,
        max_scale: float,
        ema_beta: float = 0.25,
        prevalence_alpha: float = 1.0,
        confidence_samples: int = 32,
    ) -> dict[str, float]:
        source = self.window_tags or self.total_tags
        total_count = sum(stats.count for tag, stats in source.items() if tag not in EXCLUDED_FAILURE_TAGS)
        if total_count <= 0:
            return dict(base_weights)

        ema_beta = min(1.0, max(0.0, float(ema_beta)))
        confidence_samples = max(1, int(confidence_samples))
        tag_set = set(base_weights) | set(source) | set(self.dynamic_tag_weights)

        updated: dict[str, float] = {}
        for tag in tag_set:
            base = float(base_weights.get(tag, 1.0))
            if tag in EXCLUDED_FAILURE_TAGS:
                target = 1.0
            elif tag == PASS_TAG:
                target = base
            else:
                stats = source.get(tag, CapabilityStats())
                prevalence = float(stats.count) / float(total_count)
                stagnation = 1.0 - stats.reward_mean
                confidence = min(1.0, float(stats.count) / float(confidence_samples))
                target = base * (1.0 + prevalence_alpha * prevalence * stagnation * confidence)
            target = min(max_scale, max(min_scale, target))
            previous = float(self.dynamic_tag_weights.get(tag, base))
            updated[tag] = (1.0 - ema_beta) * previous + ema_beta * target

        self.dynamic_tag_weights = updated
        return dict(updated)

    def task_weight_targets(
        self,
        *,
        min_weight: float,
        max_weight: float,
        alpha: float,
        min_samples: int,
    ) -> dict[str, float]:
        targets: dict[str, float] = {}
        source = self.window_category_tags or self.total_category_tags
        for task_id, task in self.tasks.items():
            tag = str(task.get("last_tag") or UNKNOWN_TAG)
            category = str(task.get("category") or "_none")
            stats = source.get(pack_category_tag(category, tag))
            if stats is None or stats.count < min_samples:
                continue
            if tag == PASS_TAG or tag in EXCLUDED_FAILURE_TAGS:
                target = 1.0
            else:
                confidence = min(1.0, float(stats.count) / float(max(1, min_samples * 4)))
                pressure = (1.0 - stats.reward_mean) * confidence
                target = 1.0 + float(alpha) * pressure
            targets[str(task_id)] = min(max_weight, max(min_weight, target))
        return targets

    def snapshot(self, *, window: bool = True) -> dict[str, Any]:
        categories = self.window_categories if window else self.total_categories
        tags = self.window_tags if window else self.total_tags
        category_tags = self.window_category_tags if window else self.total_category_tags
        turn_positions = self.window_turn_positions if window else self.total_turn_positions
        return {
            "epoch": self.last_epoch,
            "global_step": self.last_step,
            "categories": {key: stats.snapshot() for key, stats in categories.items()},
            "tags": {key: stats.snapshot() for key, stats in tags.items()},
            "category_tags": {key: stats.snapshot() for key, stats in category_tags.items()},
            "turn_positions": {key: stats.snapshot() for key, stats in turn_positions.items()},
            "tasks": self.tasks,
            "dynamic_tag_weights": dict(self.dynamic_tag_weights),
        }

    def metrics(self, prefix: str = "tocf/state") -> dict[str, float]:
        metrics: dict[str, float] = {
            f"{prefix}/observe_count": float(self.observe_count),
            f"{prefix}/num_tasks": float(len(self.tasks)),
            f"{prefix}/num_tags": float(len(self.total_tags)),
            f"{prefix}/num_category_tags": float(len(self.total_category_tags)),
        }
        for tag, weight in self.dynamic_tag_weights.items():
            metrics[f"{prefix}/dynamic_tag_weight/{tag}"] = float(weight)
        return metrics

    def reset_window(self) -> None:
        self.window_categories = {}
        self.window_tags = {}
        self.window_category_tags = {}
        self.window_turn_positions = {}

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_categories": self._stats_to_dict(self.total_categories),
            "window_categories": self._stats_to_dict(self.window_categories),
            "total_tags": self._stats_to_dict(self.total_tags),
            "window_tags": self._stats_to_dict(self.window_tags),
            "total_category_tags": self._stats_to_dict(self.total_category_tags),
            "window_category_tags": self._stats_to_dict(self.window_category_tags),
            "total_turn_positions": self._stats_to_dict(self.total_turn_positions),
            "window_turn_positions": self._stats_to_dict(self.window_turn_positions),
            "tasks": self.tasks,
            "dynamic_tag_weights": self.dynamic_tag_weights,
            "last_epoch": self.last_epoch,
            "last_step": self.last_step,
            "observe_count": self.observe_count,
        }

    def _load_dict(self, raw: dict[str, Any]) -> None:
        self.total_categories = self._stats_from_dict(raw.get("total_categories"))
        self.window_categories = self._stats_from_dict(raw.get("window_categories"))
        self.total_tags = self._stats_from_dict(raw.get("total_tags"))
        self.window_tags = self._stats_from_dict(raw.get("window_tags"))
        self.total_category_tags = self._stats_from_dict(raw.get("total_category_tags"))
        self.window_category_tags = self._stats_from_dict(raw.get("window_category_tags"))
        self.total_turn_positions = self._stats_from_dict(raw.get("total_turn_positions"))
        self.window_turn_positions = self._stats_from_dict(raw.get("window_turn_positions"))
        self.tasks = dict(raw.get("tasks") or {})
        self.dynamic_tag_weights = {
            str(k): float(v) for k, v in dict(raw.get("dynamic_tag_weights") or {}).items()
        }
        self.last_epoch = raw.get("last_epoch")
        self.last_step = raw.get("last_step")
        self.observe_count = int(raw.get("observe_count", 0) or 0)

    @staticmethod
    def _stats_to_dict(source: dict[str, CapabilityStats]) -> dict[str, dict[str, float | int]]:
        return {str(key): stats.to_dict() for key, stats in source.items()}

    @staticmethod
    def _stats_from_dict(raw: dict[str, Any] | None) -> dict[str, CapabilityStats]:
        return {str(key): CapabilityStats.from_dict(value) for key, value in dict(raw or {}).items()}

    def save(self, path: str | None = None) -> None:
        target = path or self.path
        if not target:
            return
        os.makedirs(os.path.dirname(target) or ".", exist_ok=True)
        with open(target, "w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, ensure_ascii=False, indent=2)
