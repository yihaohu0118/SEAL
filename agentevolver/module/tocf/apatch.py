"""A-Patch: Tag-Aware Advantage Weighting for BFCL.

Each rollout is annotated with per-turn failure tags (from BfclDenseEnvGrader's
``bfcl_dense_progress_info``). Tags indicate *why* a turn failed, not just that
it failed.  A-Patch multiplies advantages by a per-trajectory scale derived
from the tag distribution, amplifying the learning signal for the most
actionable failure modes.

Tag vocabulary (from multi_turn_progress.py):
    pass                        turn matched GT state + response
    correct_abstention          irrelevance turn: model correctly abstained
    spurious_tool_call          irrelevance turn: model wrongly called tools
    empty_turn_model_response   scorable turn: model should have called, didn't
    state_mismatch              executed tools but final state was wrong
    instance_mismatch           wrong instance set was used
    response_mismatch           state OK but response text failed
    checker_error               internal checker exception (excluded)
    gt_error                    GT execution failed (excluded)

Weighting formula for sample i in a GRPO group::

    fractions = {tag: count / len(tags) for tag, count in tag_counts.items()}
    # Exclude noisy tags that do not reflect model behaviour.
    exclude = {"checker_error", "gt_error"}
    scale_i = clamp(
        sum(fractions[t] * weight[t] for t in fractions if t not in exclude),
        min_scale, max_scale
    )
    advantages_i *= scale_i

``scale_i`` defaults to ``1.0`` when no failure tags are available (e.g.
single-turn tasks, or non-BFCL environments).

Only samples where ``extras["bfcl_failure_tags"]`` is present are scaled;
all others pass through unchanged, so enabling A-Patch on a mixed-env
dataset is safe.

Config path (all under ``tocf.advantage.apatch``):
    enable: bool (default false)
    min_scale: float (default 0.5)
    max_scale: float (default 2.0)
    apply_to: "all" | "positive_advantage" | "negative_advantage"
              (default "all")
    budget:
        enable: bool (default false)
        min_scale: float (default min_scale)
        max_scale: float (default max_scale)
        strength: float (default 1.0)
        normalize_mean: bool (default true)
        target_mean: float (default 1.0)
    metric_prefix: str (default "tocf/apatch")
    tag_weights:
        spurious_tool_call: 2.0
        correct_abstention: 1.8
        empty_turn_model_response: 1.5
        state_mismatch: 1.0
        response_mismatch: 0.9
        instance_mismatch: 0.6
        pass: 1.0
        (anything not listed defaults to 1.0)
"""
from __future__ import annotations

from typing import Any, Dict, List, Tuple

import torch

from agentevolver.module.tocf.state import TOCFCapabilityState

_DEFAULT_TAG_WEIGHTS: Dict[str, float] = {
    # ── actionable, high-signal ──────────────────────────────────────────────
    # spurious_tool_call: miss_func failure mode. The model called tools on an
    # irrelevance turn where abstaining was the only correct action. This is
    # the sharpest failure signal in BFCL: the correct action (do nothing) is
    # unambiguous.
    "spurious_tool_call": 2.0,
    # correct_abstention: the model DID correctly abstain on an irrelevance
    # turn. Amplify the positive advantage so the model keeps doing this.
    "correct_abstention": 1.8,
    # empty_turn_model_response: the model failed to produce any tool call on
    # a scorable turn. Also a clear action gap — the model froze instead of
    # acting.
    "empty_turn_model_response": 1.5,
    # ── medium-signal ────────────────────────────────────────────────────────
    "state_mismatch": 1.0,
    "response_mismatch": 0.9,
    # ── low-signal (causally ambiguous) ─────────────────────────────────────
    "instance_mismatch": 0.6,
    # ── pass (already correct) ──────────────────────────────────────────────
    "pass": 1.0,
    # checker_error / gt_error: excluded from weighting (see formula above)
}

# Tags to exclude from the weighting sum (they reflect env/scorer issues, not
# model behaviour).
_EXCLUDE_TAGS = frozenset({"checker_error", "gt_error"})


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


def apatch_enabled(config: Any) -> bool:
    """Return True iff A-Patch advantage weighting is enabled."""
    tocf_on = bool(_cfg_get(config, "tocf.enable", False))
    apatch_on = bool(_cfg_get(config, "tocf.advantage.apatch.enable", False))
    return tocf_on and apatch_on


def _apatch_cfg(config: Any) -> Any:
    return _cfg_get(config, "tocf.advantage.apatch", {}) or {}


def _tag_weight(cfg_tag_weights: Any, tag: str) -> float:
    """Resolve per-tag weight: config overrides default; missing → 1.0."""
    default = _DEFAULT_TAG_WEIGHTS.get(tag, 1.0)
    if not cfg_tag_weights:
        return default
    if isinstance(cfg_tag_weights, dict):
        return float(cfg_tag_weights.get(tag, default))
    return float(getattr(cfg_tag_weights, tag, default))


def _resolved_tag_weights(cfg_tag_weights: Any) -> Dict[str, float]:
    tags = set(_DEFAULT_TAG_WEIGHTS)
    if isinstance(cfg_tag_weights, dict):
        tags.update(str(k) for k in cfg_tag_weights)
    return {tag: _tag_weight(cfg_tag_weights, tag) for tag in tags}


def _trajectory_scale(
    failure_tags: List[str],
    cfg_tag_weights: Any,
    min_scale: float,
    max_scale: float,
) -> float:
    """Compute per-trajectory advantage scale from its failure-tag list.

    Tags in ``_EXCLUDE_TAGS`` are excluded from the weighted mean to avoid
    letting scorer exceptions drive the gradient signal.
    """
    effective = [t for t in failure_tags if t not in _EXCLUDE_TAGS]
    if not effective:
        return 1.0
    total = float(len(effective))
    raw_scale = sum(
        _tag_weight(cfg_tag_weights, t) for t in effective
    ) / total
    return float(min(max_scale, max(min_scale, raw_scale)))


def _apply_scale_budget(
    scales: torch.Tensor,
    active_indices: List[int],
    cfg: Any,
    fallback_min_scale: float,
    fallback_max_scale: float,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Optionally constrain A-Patch to a fixed per-batch gradient budget.

    In coupled T+A runs, T-Patch already changes which tasks appear in the
    batch.  If A-Patch then freely amplifies the same diagnosed failures, the
    two modules can multiply the same signal and destabilize GRPO.  This
    budget keeps A-Patch as a residual tag-level signal: scales are softened,
    clamped, and optionally mean-normalized over tagged samples.
    """
    budget_cfg = _cfg_get(cfg, "budget", {}) or {}
    if not bool(_cfg_get(budget_cfg, "enable", False)) or not active_indices:
        return scales, {
            "budget_enabled": 0.0,
            "budget_mean_before": 1.0,
            "budget_mean_after": 1.0,
        }

    idx = torch.tensor(active_indices, dtype=torch.long, device=scales.device)
    selected = scales.index_select(0, idx)
    mean_before = float(selected.mean().detach().cpu()) if selected.numel() else 1.0

    strength = float(_cfg_get(budget_cfg, "strength", 1.0) or 1.0)
    strength = max(0.0, strength)
    budget_min = float(_cfg_get(budget_cfg, "min_scale", fallback_min_scale))
    budget_max = float(_cfg_get(budget_cfg, "max_scale", fallback_max_scale))
    if budget_min > budget_max:
        budget_min, budget_max = budget_max, budget_min

    selected = 1.0 + strength * (selected - 1.0)
    selected = torch.clamp(selected, min=budget_min, max=budget_max)

    normalize = bool(_cfg_get(budget_cfg, "normalize_mean", True))
    target_mean = float(_cfg_get(budget_cfg, "target_mean", 1.0) or 1.0)
    if normalize and selected.numel():
        denom = selected.mean().clamp_min(1e-6)
        selected = selected * (target_mean / denom)
        selected = torch.clamp(selected, min=budget_min, max=budget_max)

    mean_after = float(selected.mean().detach().cpu()) if selected.numel() else 1.0
    scales = scales.clone()
    scales.index_copy_(0, idx, selected)
    return scales, {
        "budget_enabled": 1.0,
        "budget_mean_before": mean_before,
        "budget_mean_after": mean_after,
    }


def apply_apatch_advantage_weighting(
    batch: Any,
    config: Any,
    env_type: str | None = None,
    capability_state: TOCFCapabilityState | None = None,
    global_step: int | None = None,
) -> Tuple[Any, Dict[str, float]]:
    """Apply A-Patch tag-aware advantage scaling in-place on the batch.

    Args:
        batch:     DataProto batch (must have ``batch["advantages"]`` and
                   ``non_tensor_batch["extras"]``).
        config:    Hydra/OmegaConf training config.
        env_type:  The active env type string (e.g. ``"bfcl"``).  Only
                   samples that carry ``bfcl_failure_tags`` in their extras
                   are scaled; others are left unchanged.

    Returns:
        (batch, metrics_dict)
    """
    if not apatch_enabled(config):
        return batch, {}

    cfg = _apatch_cfg(config)
    min_scale = float(_cfg_get(cfg, "min_scale", 0.5))
    max_scale = float(_cfg_get(cfg, "max_scale", 2.0))
    apply_to = str(_cfg_get(cfg, "apply_to", "all") or "all").lower()
    prefix = str(_cfg_get(cfg, "metric_prefix", "tocf/apatch") or "tocf/apatch")
    cfg_tag_weights = _cfg_get(cfg, "tag_weights", None)
    warmup_steps = int(_cfg_get(cfg, "warmup_steps", 0) or 0)

    if global_step is not None and global_step < warmup_steps:
        return batch, {
            f"{prefix}/enabled": 1.0,
            f"{prefix}/warmup_skipped": 1.0,
            f"{prefix}/scaled_count": 0.0,
        }

    active_tag_weights = _resolved_tag_weights(cfg_tag_weights)
    dynamic_cfg = _cfg_get(cfg, "dynamic", {}) or {}
    dynamic_enabled = bool(_cfg_get(dynamic_cfg, "enable", capability_state is not None))
    if dynamic_enabled and capability_state is not None:
        active_tag_weights = capability_state.update_dynamic_tag_weights(
            active_tag_weights,
            min_scale=min_scale,
            max_scale=max_scale,
            ema_beta=float(_cfg_get(dynamic_cfg, "ema_beta", 0.25) or 0.25),
            prevalence_alpha=float(_cfg_get(dynamic_cfg, "prevalence_alpha", 1.0) or 1.0),
            confidence_samples=int(_cfg_get(dynamic_cfg, "confidence_samples", 32) or 32),
        )

    extras = batch.non_tensor_batch.get("extras", None)
    if extras is None:
        return batch, {f"{prefix}/enabled": 1.0, f"{prefix}/scaled_count": 0.0}

    advantages: torch.Tensor = batch.batch["advantages"]
    n_samples = advantages.shape[0]

    scales = torch.ones(n_samples, dtype=advantages.dtype, device=advantages.device)
    scaled_count = 0
    scale_sum = 0.0
    tag_count_total: Dict[str, int] = {}
    missing_tag_count = 0
    active_indices: List[int] = []

    for idx in range(n_samples):
        extra = extras[idx] if idx < len(extras) else {}
        if not isinstance(extra, dict):
            continue
        failure_tags: List[str] = extra.get("bfcl_failure_tags") or []
        if not failure_tags:
            missing_tag_count += 1
            continue

        scale = _trajectory_scale(failure_tags, active_tag_weights, min_scale, max_scale)
        scales[idx] = scale
        scaled_count += 1
        scale_sum += scale
        active_indices.append(idx)

        for t in failure_tags:
            tag_count_total[t] = tag_count_total.get(t, 0) + 1

    scales, budget_metrics = _apply_scale_budget(
        scales,
        active_indices,
        cfg,
        fallback_min_scale=min_scale,
        fallback_max_scale=max_scale,
    )

    # Apply scaling, respecting the ``apply_to`` mask.
    if apply_to == "positive_advantage":
        mask = (advantages > 0).float()
        advantages.mul_(1.0 + (scales - 1.0).unsqueeze(-1) * mask)
    elif apply_to == "negative_advantage":
        mask = (advantages < 0).float()
        advantages.mul_(1.0 + (scales - 1.0).unsqueeze(-1) * mask)
    else:  # "all"
        advantages.mul_(scales.unsqueeze(-1))

    metrics: Dict[str, float] = {
        f"{prefix}/enabled": 1.0,
        f"{prefix}/scaled_count": float(scaled_count),
        f"{prefix}/missing_tag_count": float(missing_tag_count),
        f"{prefix}/mean_scale": scale_sum / scaled_count if scaled_count else 1.0,
        f"{prefix}/budget_enabled": budget_metrics["budget_enabled"],
        f"{prefix}/budget_mean_before": budget_metrics["budget_mean_before"],
        f"{prefix}/budget_mean_after": budget_metrics["budget_mean_after"],
        f"{prefix}/dynamic_enabled": 1.0 if dynamic_enabled and capability_state is not None else 0.0,
    }
    total_tags = sum(tag_count_total.values())
    for tag, cnt in tag_count_total.items():
        metrics[f"{prefix}/tag_frac/{tag}"] = float(cnt) / float(total_tags) if total_tags else 0.0
    for tag, weight in active_tag_weights.items():
        metrics[f"{prefix}/tag_weight/{tag}"] = float(weight)

    return batch, metrics
