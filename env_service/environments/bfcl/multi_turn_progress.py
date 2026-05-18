"""Per-user-turn progress scorer for BFCL multi-turn trajectories.

The official BFCL ``multi_turn_checker`` loops through user turns and
evaluates ``state_checker`` + ``response_checker`` per turn, but it
early-returns at the first failure, giving a binary pass/fail signal.

For dense RL training we want a *progress* signal that matches the same
checker semantics but does NOT early-return. We want to know, out of the
user turns that carry a ground-truth program, how many the model actually
landed on. This is exactly the scheme used in EnvTuning
(Bytedance/SEAL): ``progress = passed_turns / scorable_turns``.

This module implements that checker. It deliberately re-uses BFCL's own
``execute_multi_turn_func_call``, ``state_checker`` and ``response_checker``,
so the per-turn 0/1 it emits is *identically aligned* with the official
grade. No process proxies, no hand-crafted shaping.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from bfcl_eval.eval_checker.multi_turn_eval.multi_turn_checker import (
    response_checker,
    state_checker,
)
from bfcl_eval.eval_checker.multi_turn_eval.multi_turn_utils import (
    execute_multi_turn_func_call,
    is_empty_execute_response,
)


def _decode_model_turns(
    handler,
    multi_turn_model_result_list: List[List[str]],
) -> List[List[List[str]]]:
    """Decode raw model output strings into executable function-call lists.

    Mirrors ``_evaluate_single_multi_turn_entry`` in BFCL's eval_runner so
    that the progress score is computed from the same decoded trajectory
    the binary runner sees.
    """
    decoded: List[List[List[str]]] = []
    for single_turn_model_result_list in multi_turn_model_result_list:
        single_turn_decoded: List[List[str]] = []
        for model_result_item in single_turn_model_result_list:
            try:
                decoded_result: List[str] = handler.decode_execute(
                    model_result_item, has_tool_call_tag=False
                )
                if is_empty_execute_response(decoded_result):
                    continue
                single_turn_decoded.append(decoded_result)
            except Exception:
                continue
        decoded.append(single_turn_decoded)
    return decoded


def compute_per_turn_progress(
    handler,
    multi_turn_model_result_list: List[List[str]],
    multi_turn_ground_truth_list: List[List[str]],
    test_entry: Dict[str, Any],
    test_category: str,
    model_name: str,
) -> Dict[str, Any]:
    """Score a multi-turn trajectory at per-user-turn granularity.

    Returns a dict with keys:
      - ``progress``: float in [0, 1]. Fraction of scorable user turns passed.
        Irrelevance / empty-GT turns are EXCLUDED from this metric (legacy
        AgentEvolver semantics, kept for backward compatibility).
      - ``progress_with_irrelevance``: float in [0, 1]. T3RL-style metric
        that ALSO scores irrelevance turns: 1.0 if the model correctly
        abstained (no decoded tool calls), 0.0 if it wrongly attempted
        tool calls. This diagnostic signal directly measures the correct
        behaviour on ``multi_turn_miss_func`` turns.
      - ``per_turn_valid``: list[Optional[bool]]. One entry per GT turn.
          * True  -> model matched GT state and response at this turn
          * False -> model failed this turn (or was force-terminated before)
          * None  -> turn has no GT program (irrelevance-style turns).
                     Excluded from ``progress`` but scored in
                     ``progress_with_irrelevance`` via ``irrelevance_per_turn``.
      - ``irrelevance_per_turn``: list[Optional[float]]. Parallel to
        ``per_turn_valid``. ``None`` for scorable turns, ``0.0``/``1.0``
        for irrelevance turns (T3RL semantics: 1.0 iff model abstained).
      - ``passed_turns`` / ``scorable_turns``: raw counts (for ``progress``).
      - ``passed_irrelevance_turns`` / ``irrelevance_turns``: raw counts
        for the T3RL-style irrelevance component.
      - ``terminated_early``: bool. True if the trajectory was shorter than
        the GT (all missing turns count as failures).
      - ``error``: Optional[str] if scoring could not proceed at all.

    Intentionally never raises from the inner checker calls; any exception
    is logged into ``errors_per_turn`` and that turn is counted as failed.
    """
    per_turn_valid: List[Optional[bool]] = []
    errors_per_turn: List[Optional[str]] = []
    irrelevance_per_turn: List[Optional[float]] = []
    failure_tags: List[str] = []  # A-Patch: per-turn structured failure tag

    if not isinstance(multi_turn_model_result_list, list):
        return {
            "progress": 0.0,
            "progress_with_irrelevance": 0.0,
            "per_turn_valid": [],
            "irrelevance_per_turn": [],
            "failure_tags": [],
            "passed_turns": 0,
            "scorable_turns": 0,
            "passed_irrelevance_turns": 0,
            "irrelevance_turns": 0,
            "terminated_early": True,
            "error": "model_result is not a list",
        }

    initial_config: dict = test_entry["initial_config"]
    involved_classes: list = test_entry["involved_classes"]
    test_entry_id: str = test_entry["id"]
    long_context = (
        "long_context" in test_category or "composite" in test_category
    )

    decoded_model_turns = _decode_model_turns(handler, multi_turn_model_result_list)

    # If trajectory was force-terminated, pad missing turns with empty decoded
    # lists so that those turns are graded as failures against their GT.
    num_gt_turns = len(multi_turn_ground_truth_list)
    num_model_turns = len(decoded_model_turns)
    terminated_early = num_model_turns < num_gt_turns
    if terminated_early:
        decoded_model_turns = decoded_model_turns + [
            [] for _ in range(num_gt_turns - num_model_turns)
        ]

    # Persistent execution cumulants, mirroring BFCL's own multi_turn_checker:
    # state is cumulative (each turn's calls mutate instances in place), and
    # response_checker gets the full accumulated execution trace so far.
    all_turn_model_execution_results: List[str] = []
    # BFCL executes by (model_name, test_entry_id); we re-use the handler id so
    # instance reuse across turns happens inside the multi_turn_utils cache.

    for turn_index, single_turn_gt in enumerate(multi_turn_ground_truth_list):
        single_turn_model = decoded_model_turns[turn_index]

        # Run model calls for this turn, even when GT is empty -- we still
        # need to mutate state so subsequent turns' state check is meaningful.
        single_turn_model_execution_results: List[str] = []
        model_instances: Dict[str, Any] = {}
        for single_step_model_response in single_turn_model:
            try:
                step_results, model_instances = execute_multi_turn_func_call(
                    func_call_list=single_step_model_response,
                    initial_config=initial_config,
                    involved_classes=involved_classes,
                    model_name=model_name,
                    test_entry_id=test_entry_id,
                    long_context=long_context,
                    is_evaL_run=True,
                )
                single_turn_model_execution_results.extend(step_results)
            except Exception as e:
                errors_per_turn.append(f"model_exec_error: {e}")
                break

        # Always execute GT for bookkeeping / response comparison, even when
        # this turn's GT is empty (no-op execution, still mutates the GT copy).
        try:
            single_turn_gt_execution_results, gt_instances = (
                execute_multi_turn_func_call(
                    func_call_list=single_turn_gt,
                    initial_config=initial_config,
                    involved_classes=involved_classes,
                    model_name=model_name + "_ground_truth",
                    test_entry_id=test_entry_id,
                    long_context=long_context,
                    is_evaL_run=True,
                )
            )
        except Exception as e:
            per_turn_valid.append(False)
            irrelevance_per_turn.append(None)
            errors_per_turn.append(f"gt_exec_error: {e}")
            failure_tags.append("gt_error")
            all_turn_model_execution_results.extend(single_turn_model_execution_results)
            continue

        all_turn_model_execution_results.extend(single_turn_model_execution_results)

        # GT-empty turn. Legacy ``progress`` excludes it (per_turn_valid=None),
        # but we ALSO emit a T3RL-style irrelevance score to expose the
        # behaviour of *not* calling tools on multi_turn_miss_func reveal turns.
        #
        # Score mirrors t3rl/envs/bfcl_gym.py:_step_on_answer in training mode:
        #   - 1.0  iff model emitted no decoded tool calls this turn
        #   - 0.0  iff model still tried to call tools despite the empty GT
        if not single_turn_gt:
            per_turn_valid.append(None)
            model_attempted_tools = bool(single_turn_model) and not is_empty_execute_response(
                single_turn_model
            )
            irr_score = 0.0 if model_attempted_tools else 1.0
            irrelevance_per_turn.append(irr_score)
            errors_per_turn.append(None)
            failure_tags.append(
                "spurious_tool_call" if model_attempted_tools else "correct_abstention"
            )
            continue

        # Require that the model actually produced some function call for a
        # GT-nonempty turn, matching the official ``empty_turn_model_response``
        # rule from multi_turn_checker.
        if not single_turn_model or is_empty_execute_response(single_turn_model):
            per_turn_valid.append(False)
            irrelevance_per_turn.append(None)
            errors_per_turn.append("empty_turn_model_response")
            failure_tags.append("empty_turn_model_response")
            continue

        if not gt_instances or set(model_instances.keys()) != set(gt_instances.keys()):
            per_turn_valid.append(False)
            irrelevance_per_turn.append(None)
            errors_per_turn.append("instance_mismatch")
            failure_tags.append("instance_mismatch")
            continue

        try:
            sc = state_checker(model_instances, gt_instances)
            state_ok = bool(sc.get("valid", False))
        except Exception as e:
            state_ok = False
            errors_per_turn.append(f"state_checker_error: {e}")
            per_turn_valid.append(False)
            irrelevance_per_turn.append(None)
            failure_tags.append("checker_error")
            continue

        if not state_ok:
            per_turn_valid.append(False)
            irrelevance_per_turn.append(None)
            errors_per_turn.append("state_mismatch")
            failure_tags.append("state_mismatch")
            continue

        try:
            rc = response_checker(
                all_turn_model_execution_results,
                single_turn_gt_execution_results,
                turn_index,
            )
            response_ok = bool(rc.get("valid", False))
        except Exception as e:
            response_ok = False
            errors_per_turn.append(f"response_checker_error: {e}")
            per_turn_valid.append(False)
            irrelevance_per_turn.append(None)
            failure_tags.append("checker_error")
            continue

        if not response_ok:
            per_turn_valid.append(False)
            irrelevance_per_turn.append(None)
            errors_per_turn.append("response_mismatch")
            failure_tags.append("response_mismatch")
            continue

        per_turn_valid.append(True)
        irrelevance_per_turn.append(None)
        errors_per_turn.append(None)
        failure_tags.append("pass")

    passed_turns = sum(1 for v in per_turn_valid if v is True)
    scorable_turns = sum(1 for v in per_turn_valid if v is not None)
    progress = float(passed_turns) / float(scorable_turns) if scorable_turns > 0 else 0.0

    passed_irrelevance_turns = sum(
        1 for s in irrelevance_per_turn if isinstance(s, float) and s >= 1.0
    )
    irrelevance_turns = sum(1 for s in irrelevance_per_turn if s is not None)

    # T3RL-style progress: mix scorable-turn passes with irrelevance-turn
    # abstention rewards. Denominator = scorable + irrelevance turns. When
    # there are no irrelevance turns this equals ``progress`` exactly, so the
    # new metric is a strict superset of the legacy one.
    combined_passed = passed_turns + passed_irrelevance_turns
    combined_total = scorable_turns + irrelevance_turns
    progress_with_irrelevance = (
        float(combined_passed) / float(combined_total) if combined_total > 0 else 0.0
    )

    return {
        "progress": progress,
        "progress_with_irrelevance": progress_with_irrelevance,
        "per_turn_valid": per_turn_valid,
        "irrelevance_per_turn": irrelevance_per_turn,
        "failure_tags": failure_tags,
        "passed_turns": passed_turns,
        "scorable_turns": scorable_turns,
        "passed_irrelevance_turns": passed_irrelevance_turns,
        "irrelevance_turns": irrelevance_turns,
        "terminated_early": terminated_early,
        "errors_per_turn": errors_per_turn,
    }


def safe_compute_progress(
    handler,
    model_result_data: Dict[str, Any],
    possible_answer: List[Dict[str, Any]],
    prompt_entry: Dict[str, Any],
    test_category: str,
    model_name: str,
) -> Tuple[float, Optional[Dict[str, Any]]]:
    """Top-level safe wrapper used by env_handler.

    Returns ``(progress, info_dict_or_none)``. Any failure is swallowed to a
    progress of 0.0 so that a broken progress scorer never crashes the env.
    """
    try:
        if not possible_answer:
            return 0.0, {"error": "no_possible_answer"}

        # multi_turn_runner uses possible_answer[i]["ground_truth"]
        gt_list = possible_answer[0].get("ground_truth", [])
        mr_list = model_result_data.get("result", [])

        info = compute_per_turn_progress(
            handler=handler,
            multi_turn_model_result_list=mr_list,
            multi_turn_ground_truth_list=gt_list,
            test_entry=prompt_entry,
            test_category=test_category,
            model_name=model_name,
        )
        return float(info.get("progress", 0.0)), info
    except Exception as e:
        return 0.0, {"error": f"progress_exception: {e}"}
