import json
import time
from typing import Optional, cast

from loguru import logger

from agentevolver.module.agent_flow.agent_flow import AgentFlow
from agentevolver.module.agent_flow.reward_calculator import RewardCalculator


class ModifiedAgentFlow(AgentFlow):
    def __init__(self, reward_calculator: Optional[RewardCalculator] = None, **kwargs):
        super().__init__(reward_calculator, **kwargs)
