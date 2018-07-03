"""Agent that implements explore behavior."""

from swarms.lib.agent import Agent
from swarms.sbehaviors import (
    RandomWalk, Move
    )

import py_trees
import numpy as np


class SwarmAgentRandomWalk(Agent):
    """Swarm agent.

    An minimalistic behavior tree for swarm agent
    implementing carry and drop behavior.
    """

    def __init__(self, name, model):
        """Initialize the agent methods."""
        super().__init__(name, model)
        self.location = ()

        self.direction = model.random.rand() * (2 * np.pi)
        self.speed = 2
        self.radius = 3

        self.moveable = True
        self.shared_content = dict()

        self.shared_content['Hub'] = {model.hub}

        # Just checking the randomwalk behavior with many agent
        # how much they explore the environment

        r1 = RandomWalk('1')
        r1.setup(0, self, None)

        m1 = Move('2')
        m1.setup(0, self, None)

        randseq = py_trees.composites.Sequence('RSequence')
        randseq.add_children([r1, m1])
        self.behaviour_tree = py_trees.trees.BehaviourTree(randseq)

    def step(self):
        """Execute the behavior."""
        self.behaviour_tree.tick()

    def advance(self):
        """Simulteneous execution."""
        pass
