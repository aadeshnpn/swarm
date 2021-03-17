"""Derived agent class."""

from swarms.lib.agent import Agent
import numpy as np
from swarms.utils.bt import BTConstruct

# from swarms.utils.results import Results
# from py_trees import Status
# import copy
import py_trees

from swarms.behaviors.sbehaviors import (
    NeighbourObjects, IsVisitedBefore,
    IsCarrying, IsInPartialAttached, ObjectsOnGrid
    )

from swarms.behaviors.scbehaviors import (
    CompositeDrop, CompositeSingleCarry, MoveTowards,
    Explore, CompositeDropPartial, CompositeMultipleCarry, AgentDead
)

# import py_trees


class SimForgAgent(Agent):
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

        # self.bt.xmlstring = xmlstring
        # self.bt.construct()

        # Drop branch
        dseq = py_trees.composites.Sequence('DSequence')
        iscarrying = IsCarrying('IsCarrying_Food')
        iscarrying.setup(0, self, 'Food')

        neighhub = NeighbourObjects('NeighbourObjects_Hub')
        neighhub.setup(0, self, 'Hub')

        notneighhub = py_trees.meta.inverter(NeighbourObjects)(
            'NeighbourObjects_Hub')
        notneighhub.setup(0, self, 'Hub')

        drop = CompositeDrop('CompositeDrop_Food')
        drop.setup(0, self, 'Food')

        dseq.add_children([neighhub, drop])

        # ## Obstacles and Trap
        # neighobst = ObjectsOnGrid('ObjectsOnGrid_Obstacles')
        # neighobst.setup(0, self, 'Obstacles')

        # neightrap = ObjectsOnGrid('NeighbourObjects_Traps')
        # neightrap.setup(0, self, 'Traps')

        adead = AgentDead('Adead')
        adead.setup(0, self)

        # Carry branch
        cseq = py_trees.composites.Sequence('CSequence')

        neighsite = NeighbourObjects('NeighbourObjects_Sites')
        neighsite.setup(0, self, 'Sites')

        neighfood = NeighbourObjects('NeighbourObjects_Food')
        neighfood.setup(0, self, 'Food')

        invcarrying = py_trees.meta.inverter(IsCarrying)('IsCarrying_Food')
        invcarrying.setup(0, self, 'Food')

        carry = CompositeSingleCarry('CompositeSingleCarry_Food')
        carry.setup(0, self, 'Food')

        cseq.add_children([neighsite, neighfood, invcarrying, carry])

        # Locomotion branch

        # Move to site
        siteseq = py_trees.composites.Sequence('SiteSeq')

        sitefound = IsVisitedBefore('IsVisitedBefore_Sites')
        sitefound.setup(0, self, 'Sites')

        gotosite = MoveTowards('MoveTowards_Sites')
        gotosite.setup(0, self, 'Sites')

        # siteseq.add_children([neighobst, neightrap, sitefound, invcarrying, gotosite])        
        siteseq.add_children([sitefound, invcarrying, gotosite, adead])
        # siteseq.add_children([invcarrying])

        # Move to hub
        hubseq = py_trees.composites.Sequence('HubSeq')

        gotohub = MoveTowards('MoveTowards_Hub')
        gotohub.setup(0, self, 'Hub')

        # hubseq.add_children([neighobst, neightrap, iscarrying, gotohub])
        hubseq.add_children([iscarrying, gotohub, adead])        

        sitenotfound = py_trees.meta.inverter(IsVisitedBefore)(
            'IsVisitedBefore_Sites')
        sitenotfound.setup(0, self, 'Sites')

        explore = Explore('Explore')
        explore.setup(0, self)

        randwalk = py_trees.composites.Sequence('Randwalk')
        # randwalk.add_children([neighobst, neightrap, sitenotfound, explore])
        randwalk.add_children([sitenotfound, explore])        

        locoselect = py_trees.composites.Selector('Move')
        locoselect.add_children([siteseq, hubseq, explore])
        # locoselect.add_children([hubseq, randwalk])

        select = py_trees.composites.Selector('Main')

        select.add_children([dseq, cseq, locoselect])

        self.behaviour_tree = py_trees.trees.BehaviourTree(select)

        # py_trees.display.render_dot_tree(
        #     self.behaviour_tree.root, name=model.pname + '/forgehc')
        # py_trees.logging.level = py_trees.logging.Level.DEBUG
        # py_trees.display.print_ascii_tree(select)

    def step(self):
        # self.bt.behaviour_tree.tick()
        self.behaviour_tree.tick()

    def advance(self):
        pass
