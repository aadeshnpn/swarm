"""Derived agent class."""

from swarms.lib.agent import Agent
import numpy as np
import py_trees
from swarms.sbehaviors import (
    NeighbourObjects, IsCarryable,
    IsMultipleCarry, IsInPartialAttached,
    InitiateMultipleCarry, IsEnoughStrengthToCarry,
    GoTo, Move, IsDropable, DropPartial
    )


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
        # self.weight = 5
        self.shared_content = dict()
        self.signals = []

        # Initialize the behavior trees with a Behavior tree
        self.behaviour_tree = self.create_bt()

    def create_bt(self):
        """Create behaviors tree."""
        self.shared_content['Hub'] = [self.model.hub]

        carryroot = py_trees.composites.Sequence("Sequence")
        lowest = NeighbourObjects('0')
        lowest.setup(0, self, 'Derbis')

        low = IsCarryable('1')
        low.setup(0, self, 'Derbis')

        medium = IsMultipleCarry('2')
        medium.setup(0, self, 'Derbis')

        r1Sequence = py_trees.composites.Sequence("R1Sequence")
        r1Selector = py_trees.composites.Selector("R1Selector")
        mainSelector = py_trees.composites.Selector("MainSelector")

        high1 = IsInPartialAttached('3')
        high1.setup(0, self, 'Derbis')

        high2 = InitiateMultipleCarry('4')
        high2.setup(0, self, 'Derbis')

        high3 = IsEnoughStrengthToCarry('5')
        high3.setup(0, self, 'Derbis')

        high4 = GoTo('6')
        high4.setup(0, self, 'Hub')

        high5 = Move('7')
        high5.setup(0, self)

        nearHub1 = NeighbourObjects('13')
        nearHub1.setup(0, self, 'Hub')

        r1Selector.add_children([high1, high2])
        r1Sequence.add_children([medium, r1Selector])

        carryroot.add_children([lowest, low, r1Sequence])
        mainSelector.add_children([nearHub1, carryroot])

        # Adding new sub-tree for drop logic for multiple carry
        droproot = py_trees.composites.Selector("DropSelector")
        moveSequence = py_trees.composites.Sequence("MoveSequence")
        dropSequence = py_trees.composites.Sequence("DropSequence")

        nearHub = py_trees.meta.inverter(NeighbourObjects)('8')
        nearHub.setup(0, self, 'Hub')

        moveSequence.add_children([nearHub, high3, high4, high5])

        high6 = NeighbourObjects('9')
        high6.setup(0, self, 'Hub')

        high7 = IsDropable('10')
        high7.setup(0, self, 'Hub')

        high8 = IsInPartialAttached('11')
        high8.setup(0, self, 'Derbis')

        high9 = DropPartial('12')
        high9.setup(0, self, 'Derbis')

        dropSequence.add_children([high6, high7, high8, high9])
        droproot.add_children([moveSequence, dropSequence])

        root = py_trees.composites.Sequence("Root")
        root.add_children([mainSelector, droproot])

        behaviour_tree = py_trees.trees.BehaviourTree(root)
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
