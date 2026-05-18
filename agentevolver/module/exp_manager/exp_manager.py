from dataclasses import dataclass, field
from typing import List, Literal, Tuple

from omegaconf import DictConfig

from agentevolver.schema.task import Task


@dataclass
class TaskExpConfig:
    add_exp: List[bool]
    train_mode: str = "discard"


@dataclass
class TrajExpConfig:
    add_exp: bool = False
    train_mode: str = "discard"
    task_id: str = ""
    data_id: str = ""
    rollout_id: str = ""
    query: str = ""
    mode: str = "sample"
    experience_list: List[str] = field(default_factory=list)


class ExperienceManager:
    """BFCL release no-op compatibility layer.

    SEAL does not use external experience retrieval or memory summarization.
    The trainer still expects per-task rollout configs, so this class returns
    deterministic no-experience settings for every rollout.
    """

    def __init__(self, config: DictConfig):
        self.config = config
        self.rollout_config = config.actor_rollout_ref.rollout

    def get_complete_exp_configs(
        self, tasks: List[Task], mode: Literal["sample", "validate"]
    ) -> List[TaskExpConfig]:
        rollout_n = (
            self.rollout_config.val_kwargs.n
            if mode == "validate"
            else self.rollout_config.n
        )
        return [
            TaskExpConfig(add_exp=[False] * int(rollout_n), train_mode="discard")
            for _ in tasks
        ]


class ExperienceWorker:
    """No-op rollout/training context manager for the BFCL-only release."""

    def __init__(self, config: DictConfig):
        self.config = config

    def manage_rollout_context(
        self, init_messages: List[dict], traj_exp_config: TrajExpConfig
    ) -> Tuple[List[dict], TrajExpConfig]:
        traj_exp_config.add_exp = False
        traj_exp_config.experience_list = []
        return init_messages, traj_exp_config

    def manage_training_context(self, message: str, metadata_config: dict) -> Tuple[str, str]:
        return "", message
