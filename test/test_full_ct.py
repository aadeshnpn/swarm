from swarms.lib.agent import Agent
from swarms.lib.objects import Sites, Food, Hub
from swarms.lib.model import Model
from swarms.lib.time import SimultaneousActivation
from swarms.lib.space import Grid
from unittest import TestCase
import py_trees

import numpy as np

from swarms.behaviors.sbehaviors import (
    NeighbourObjects, IsVisitedBefore, IsInPartialAttached)

from swarms.behaviors.scbehaviors import (
    MoveTowards, CompositeDropPartial, CompositeMultipleCarry,
    Explore)


class SwarmMultipleCarry(Agent):
    """An minimalistic behavior tree for swarm agent implementing
    MultipleCarry behavior
    """
    def __init__(self, name, model):
        super().__init__(name, model)
        self.location = ()

        self.direction = model.random.rand() * (2 * np.pi)
        # self.speed = 2
        self.radius = 3

        self.moveable = True
        self.shared_content = dict()
        self.shared_content['Hub'] = {model.hub}
        self.shared_content['Sites'] = {model.site}

        # Drop branch
        dseq = py_trees.composites.Sequence('DSequence')

        iscarrying = IsInPartialAttached('8')
        iscarrying.setup(0, self, 'Food')

        # If near hub and carrying food with other agents drop
        neighhub = NeighbourObjects('2')
        neighhub.setup(0, self, 'Hub')

        drop = CompositeDropPartial('4')
        drop.setup(0, self, 'Food')

        dseq.add_children([neighhub, drop])

        # Carry branch
        cseq = py_trees.composites.Sequence('CSequence')

        # neighsite = NeighbourObjects('5')
        # neighsite.setup(0, self, 'Sites')
        neighhub = py_trees.meta.inverter(NeighbourObjects)('40')
        neighhub.setup(0, self, 'Hub')

        neighfood = NeighbourObjects('50')
        neighfood.setup(0, self, 'Food')

        invcarrying = py_trees.meta.inverter(IsInPartialAttached)('8')
        invcarrying.setup(0, self, 'Food')

        carry = CompositeMultipleCarry('9')
        carry.setup(0, self, 'Food')

        # Move to hub
        hubseq = py_trees.composites.Sequence('HubSeq')
        # If carrying something to go to hub
        gotohub = MoveTowards('10')
        gotohub.setup(0, self, 'Hub')

        hubseq.add_children([iscarrying, gotohub])

        cseq.add_children([neighhub, neighfood, carry, hubseq])

        # Locomotion branch

        # Move to site
        siteseq = py_trees.composites.Sequence('SiteSeq')
        # If site already found and not carrying anything go to site
        sitefound = IsVisitedBefore('7')
        sitefound.setup(0, self, 'Sites')

        gotosite = MoveTowards('9')
        gotosite.setup(0, self, 'Sites')

        siteseq.add_children([sitefound, invcarrying, gotosite])

        # Do Random walk
        explore = Explore('12')
        explore.setup(0, self)

        randwalk = py_trees.composites.Sequence('Randwalk')
        randwalk.add_children([explore])
        # Try to got to site and hub if not explore
        locoselect = py_trees.composites.Selector('Move')
        locoselect.add_children([iscarrying, siteseq, explore])
        # First try to drop then collect or explore
        select = py_trees.composites.Selector('Main')
        select.add_children([dseq, cseq, locoselect])

        self.behaviour_tree = py_trees.trees.BehaviourTree(select)

        # Debugging stuffs for py_trees
        # py_trees.logging.level = py_trees.logging.Level.DEBUG
        # py_trees.display.print_ascii_tree(select)

    def step(self):
        self.behaviour_tree.tick()


class MultipleCarryModel(Model):
    """ A environment to model swarms """
    def __init__(self, N, width, height, grid=10, seed=None):
        if seed is None:
            super(MultipleCarryModel, self).__init__(seed=None)
        else:
            super(MultipleCarryModel, self).__init__(seed)

        self.num_agents = N

        self.grid = Grid(width, height, grid)

        self.schedule = SimultaneousActivation(self)

        self.thing = Food(id=1, location=(30, 30), radius=40)
        self.grid.add_object_to_grid(self.thing.location, self.thing)

        self.thing1 = Food(id=1, location=(30, 30), radius=40)
        self.grid.add_object_to_grid(self.thing1.location, self.thing1)

        self.site = Sites(id=1, location=(30, 30), radius=5, q_value=0.5)
        self.grid.add_object_to_grid(self.site.location, self.site)

        self.hub = Hub(id=1, location=(0, 0), radius=11)
        self.grid.add_object_to_grid(self.hub.location, self.hub)

        self.agent = []
        for i in range(self.num_agents):
            a = SwarmMultipleCarry(i, self)
            self.schedule.add(a)
            x = 30
            y = 30
            a.location = (x, y)
            a.direction = 2.3561944901923448
            self.grid.add_object_to_grid((x, y), a)
            self.agent.append(a)

    def step(self):
        self.schedule.step()


class TestMultipleCarry(TestCase):

    def setUp(self):
        self.environment = MultipleCarryModel(
            2, 100, 100, 10, 123)

        for i in range(20):
            print(
                i, [agent.location for agent in self.environment.agent],
                self.environment.thing1.location)
            self.environment.step()

    def tuple_round(self, loc):
        loc1 = (np.round(loc[0]), np.round(loc[1]))
        return loc1

    def test_total_food_reach_hub(self):
        grid = self.environment.grid
        # hub_loc = self.environment.hub.location
        neighbours = grid.get_neighborhood((0, 0), 5)
        food_objects = grid.get_objects_from_list_of_grid('Food', neighbours)
        self.assertEqual(2, len(food_objects))
