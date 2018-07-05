"""Agent class for single source foraging."""

from swarms.lib.agent import Agent
from swarms.behaviors.sbehaviors import (
    Move, NeighbourObjects, IsCarryable,
    IsSingleCarry, SingleCarry, IsCarrying,
    IsDropable, Drop, GoTo, IsVisitedBefore, RandomWalk
    )
from py_trees.composites import RepeatUntilFalse
import py_trees
import numpy as np


class SwarmAgentRandomSingleCarryDropReturn(Agent):
    """ An minimalistic behavior tree for swarm agent
    implementing carry and drop behavior
    """
    def __init__(self, name, model):
        super().__init__(name, model)
        self.location = ()

        self.direction = model.random.rand() * (2 * np.pi)
        self.speed = 2
        self.radius = 3

        self.moveable = True
        self.shared_content = dict()

        # self.shared_content['Hub'] = model.hub
        # self.shared_content['Sites'] = model.site
        self.shared_content['Hub'] = {model.hub}
        # root = py_trees.composites.Sequence("Sequence")
        # root = py_trees.composites.Selector('Selector')
        mseq = py_trees.composites.Sequence('MSequence')
        nseq = py_trees.composites.Sequence('NSequence')
        select = py_trees.composites.Selector('RSelector')
        carryseq = py_trees.composites.Sequence('CSequence')
        dropseq = py_trees.composites.Sequence('DSequence')

        lowest1 = py_trees.meta.inverter(NeighbourObjects)('00')
        lowest1.setup(0, self, 'Hub')

        lowest11 = NeighbourObjects('0')
        lowest11.setup(0, self, 'Sites')

        lowest = NeighbourObjects('0')
        lowest.setup(0, self, 'Food')

        low = IsCarryable('1')
        low.setup(0, self, 'Food')

        medium = IsSingleCarry('2')
        medium.setup(0, self, 'Food')

        high = SingleCarry('3')
        high.setup(0, self, 'Food')

        carryseq.add_children([lowest1, lowest11, lowest, low, medium, high])

        repeathub = RepeatUntilFalse("RepeatSeqHub")
        repeatsite = RepeatUntilFalse("RepeatSeqSite")

        high1 = py_trees.meta.inverter(NeighbourObjects)('4')
        # high1 = NeighbourObjects('4')
        high1.setup(0, self, 'Hub')

        med1 = GoTo('5')
        med1.setup(0, self, 'Hub')

        # low1 = py_trees.meta.inverter(Move)('6')
        low1 = Move('6')
        low1.setup(0, self, None)

        high2 = py_trees.meta.inverter(NeighbourObjects)('12')
        # high2 = NeighbourObjects('12')
        high2.setup(0, self, 'Sites')

        # med2 = py_trees.meta.inverter(GoTo)('13')
        med2 = GoTo('13')
        med2.setup(0, self, 'Sites')

        # low1 = py_trees.meta.inverter(Move)('6')
        low2 = Move('14')
        low2.setup(0, self, None)

        # Drop
        dropseq = py_trees.composites.Sequence('DSequence')
        c1 = IsCarrying('7')
        c1.setup(0, self, 'Food')

        d1 = IsDropable('8')
        d1.setup(0, self, 'Hub')

        d2 = Drop('9')
        d2.setup(0, self, 'Food')

        dropseq.add_children([c1, d1, d2])

        repeathub.add_children([high1, med1, low1])
        repeatsite.add_children([high2, med2, low2])

        mseq.add_children([carryseq, repeathub])
        nseq.add_children([dropseq, repeatsite])

        # For randomwalk to work the agents shouldn't know the location of Site
        v1 = py_trees.meta.inverter(IsVisitedBefore)('15')
        v1.setup(0, self, 'Sites')

        r1 = RandomWalk('16')
        r1.setup(0, self, None)

        m1 = Move('17')
        m1.setup(0, self, None)

        randseq = py_trees.composites.Sequence('RSequence')
        randseq.add_children([v1, r1, m1])

        select.add_children([nseq, mseq, randseq])

        self.behaviour_tree = py_trees.trees.BehaviourTree(select)
        # py_trees.logging.level = py_trees.logging.Level.DEBUG
        # py_trees.display.print_ascii_tree(select)

    def step(self):
        self.behaviour_tree.tick()

    def advance(self):
        pass