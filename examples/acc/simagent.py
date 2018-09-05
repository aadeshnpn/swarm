"""Derived agent class."""

from swarms.lib.agent import Agent
import numpy as np
from swarms.utils.bt import BTConstruct
# from swarms.utils.results import Results

"""
from swarms.behaviors.sbehaviors import (
    Move, NeighbourObjects, IsCarryable,
    IsSingleCarry, SingleCarry,
    IsDropable, Drop, GoTo, IsVisitedBefore, RandomWalk,
    IsCarrying
    )
from swarms.behaviors.scbehaviors import (
    CompositeDrop, CompositeSingleCarry, MoveTowards,
    Explore
)
"""

# import py_trees


class SimAgent(Agent):
    """Simulation agent.

    An minimalistic behavior tree for swarm agent
    implementing carry and drop behavior.
    """

    def __init__(self, name, model, xmlstring=None):
        super().__init__(name, model)
        self.location = ()

        self.direction = model.random.rand() * (2 * np.pi)
        self.speed = 2
        self.radius = 3

        self.moveable = True
        self.shared_content = dict()

        self.carryable = False
        # Define a BTContruct object
        self.bt = BTConstruct(None, self)

        class DummyIndividual:
            def __init__(self):
                self.phenotype = None
        dummyind = DummyIndividual()
        self.individual = [dummyind]
        self.individual[0].phenotype = xmlstring

        self.bt.xmlstring = xmlstring
        self.bt.construct()
        """
        # Drop branch
        dseq = py_trees.composites.Sequence('DSequence')
        iscarrying = IsCarrying('1')
        iscarrying.setup(0, self, 'Food')

        neighhub = NeighbourObjects('2')
        neighhub.setup(0, self, 'Hub')

        drop = CompositeDrop('4')
        drop.setup(0, self, 'Food')

        dseq.add_children([neighhub, drop])

        # Carry branch
        cseq = py_trees.composites.Sequence('CSequence')

        neighsite = NeighbourObjects('5')
        neighsite.setup(0, self, 'Sites')

        neighfood = NeighbourObjects('50')
        neighfood.setup(0, self, 'Food')

        invcarrying = py_trees.meta.inverter(IsCarrying)('8')
        invcarrying.setup(0, self, 'Food')

        carry = CompositeSingleCarry('6')
        carry.setup(0, self, 'Food')

        cseq.add_children([neighsite, neighfood, invcarrying, carry])

        # Locomotion branch

        # Move to site
        siteseq = py_trees.composites.Sequence('SiteSeq')

        sitefound = IsVisitedBefore('7')
        sitefound.setup(0, self, 'Sites')

        gotosite = MoveTowards('9')
        gotosite.setup(0, self, 'Sites')

        siteseq.add_children([sitefound, invcarrying, gotosite])

        # Move to hub
        hubseq = py_trees.composites.Sequence('HubSeq')

        gotohub = MoveTowards('10')
        gotohub.setup(0, self, 'Hub')

        hubseq.add_children([iscarrying, gotohub])

        sitenotfound = py_trees.meta.inverter(IsVisitedBefore)('11')
        sitenotfound.setup(0, self, 'Sites')

        explore = Explore('12')
        explore.setup(0, self)

        randwalk = py_trees.composites.Sequence('Randwalk')
        randwalk.add_children([explore])

        locoselect = py_trees.composites.Selector('Move')
        locoselect.add_children([siteseq, hubseq, explore])
        select = py_trees.composites.Selector('Main')

        select.add_children([dseq, cseq, locoselect])

        self.behaviour_tree = py_trees.trees.BehaviourTree(select)
        """
        # py_trees.logging.level = py_trees.logging.Level.DEBUG
        # py_trees.display.print_ascii_tree(select)

    def step(self):
        self.bt.behaviour_tree.tick()

    def advance(self):
        pass