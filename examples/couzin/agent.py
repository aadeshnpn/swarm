"""Couzin's agent.
Implementation of seminal paper
'Collective Memory and Spatial Sorting in Animal Groups'
using swarm framework.
"""

from swarms.lib.agent import Agent
import numpy as np


class CAgent(Agent):
    """
    Couzin's agent to perform four different behaviors.

    Swarm, torus, dynamic and highly parallel group.
    """

    def __init__(
        self, name, model, location, direction, rep_r, orient_r, att_r,
            blind_angle, max_turn_angle):
        """Initialize the agent."""
        super().__init__(name, model)
        self.__dict__.update(locals())

    def step(self):
        """Interaction with the environment."""
        pass
