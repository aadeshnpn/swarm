"""Derived agent class."""

from swarms.lib.agent import Agent
import numpy as np
import py_trees
from swarms.sbehaviors import RandomWalk, Move


class SwarmAgent(Agent):
    """An minimalistic swarm agent."""

    def __init__(self, name, model):
        """Initialize the agent."""
        super().__init__(name, model)
        self.location = ()

        self.direction = model.random.rand() * (2 * np.pi)

        # This variable was used for move function. Since we are adopting
        # accleration based method this
        # variable is only kept for the tests to pass
        self.speed = 2
        self.radius = 3
        self.moveable = True
        self.weight = 5
        self.shared_contents = dict()
        self.signals = []

        # Initialize the behavior trees with a Behavior tree
        self.behaviour_tree = self.create_bt()

    def create_bt(self):
        """Create behaviors tree."""
        r1 = RandomWalk('1')
        r1.setup(0, self, None)

        m1 = Move('2')
        m1.setup(0, self, None)

        randseq = py_trees.composites.Sequence('RSequence')
        randseq.add_children([r1, m1])
        behaviour_tree = py_trees.trees.BehaviourTree(randseq)
        # py_trees.logging.level = py_trees.logging.Level.DEBUG
        # py_trees.display.print_ascii_tree(randseq)
        return behaviour_tree

    # New Agent methods for behavior based robotics
    def sense(self):
        """Sense included in behavior tree."""
        pass

    def plan(self):
        """Plan not required for now."""
        pass

    # Make necessary Changes
    def step(self):
        """Need to change."""
        self.behaviour_tree.tick()
