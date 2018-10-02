"""Inherited model class."""
from swarms.lib.agent import Agent
from swarms.lib.objects import Obstacles, Hub, Debris
from swarms.lib.model import Model
from swarms.lib.time import SimultaneousActivation
from swarms.lib.space import Grid
from unittest import TestCase
import py_trees

import numpy as np


from swarms.behaviors.sbehaviors import (
    NeighbourObjects, IsCarrying, IsVisitedBefore
)

from swarms.behaviors.scbehaviors import (
    CompositeSingleCarry, CompositeDrop, Explore,
    MoveTowards
)


class NM(Agent):
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
        self.shared_content['Obstacles'] = set(model.obstacles)

        # Drop branch
        dseq = py_trees.composites.Sequence('DSequence')
        iscarrying = IsCarrying('1')
        iscarrying.setup(0, self, 'Debris')

        neighhub = NeighbourObjects('2')
        neighhub.setup(0, self, 'Obstacles')

        drop = CompositeDrop('4')
        drop.setup(0, self, 'Debris')

        dseq.add_children([neighhub, drop])

        # Carry branch
        cseq = py_trees.composites.Sequence('CSequence')

        neighsite = py_trees.meta.inverter(NeighbourObjects)('5')
        neighsite.setup(0, self, 'Obstacles')

        neighfood = NeighbourObjects('50')
        neighfood.setup(0, self, 'Debris')

        invcarrying = py_trees.meta.inverter(IsCarrying)('8')
        invcarrying.setup(0, self, 'Debris')

        carry = CompositeSingleCarry('6')
        carry.setup(0, self, 'Debris')

        cseq.add_children([neighsite, neighfood, invcarrying, carry])

        # Locomotion branch

        # Move to site
        siteseq = py_trees.composites.Sequence('SiteSeq')

        sitefound = IsVisitedBefore('7')
        sitefound.setup(0, self, 'Obstacles')

        gotosite = MoveTowards('9')
        gotosite.setup(0, self, 'Obstacles')

        siteseq.add_children([sitefound, iscarrying, gotosite])

        # Move to hub
        hubseq = py_trees.composites.Sequence('HubSeq')

        gotohub = MoveTowards('10')
        gotohub.setup(0, self, 'Hub')

        hubseq.add_children([iscarrying, gotohub])

        sitenotfound = py_trees.meta.inverter(IsVisitedBefore)('11')
        sitenotfound.setup(0, self, 'Obstacles')

        explore = Explore('12')
        explore.setup(0, self)

        randwalk = py_trees.composites.Sequence('Randwalk')
        randwalk.add_children([explore])

        locoselect = py_trees.composites.Selector('Move')
        locoselect.add_children([siteseq, explore])
        select = py_trees.composites.Selector('Main')

        select.add_children([dseq, cseq, locoselect])

        self.behaviour_tree = py_trees.trees.BehaviourTree(select)

        # Debugging stuffs for py_trees
        # py_trees.logging.level = py_trees.logging.Level.DEBUG
        # py_trees.display.print_ascii_tree(select)

    def step(self):
        self.behaviour_tree.tick()


class NMModel(Model):
    """ A environment to model swarms """
    def __init__(self, N, width, height, grid=10, seed=None):
        if seed is None:
            super(NMModel, self).__init__(seed=None)
        else:
            super(NMModel, self).__init__(seed)

        self.num_agents = N

        self.grid = Grid(width, height, grid)

        self.schedule = SimultaneousActivation(self)

        self.hub = Hub(id=1, location=(0, 0), radius=11)
        self.grid.add_object_to_grid(self.hub.location, self.hub)

        # Create a place for the agents to drop the derbis
        self.obstacles = []
        try:
            for i in range(4):
                dx, dy = self.random.randint(5, 10, 2)
                dx = self.hub.location[0] + 25 + dx
                dy = self.hub.location[1] + 25 + dy
                o = Obstacles(id=i, location=(dx, dy), radius=10)
                self.grid.add_object_to_grid(o.location, o)
                self.obstacles.append(o)
        except AttributeError:
            pass

        self.agent = []
        for i in range(self.num_agents):
            a = NM(i, self)
            self.schedule.add(a)
            x = 19
            y = 10
            a.location = (x, y)
            a.direction = 2.3561944901923448
            self.grid.add_object_to_grid((x, y), a)
            self.agent.append(a)
        self.debris = []
        try:
            for i in range(self.num_agents * 10):
                dx, dy = self.random.randint(10, 20, 2)
                dx = self.hub.location[0] + dx
                dy = self.hub.location[1] + dy
                d = Debris(
                    i, location=(dx, dy),
                    radius=5)
                # print (i, dx, dy)
                d.agent_name = None
                self.grid.add_object_to_grid(d.location, d)
                self.debris.append(d)
        except KeyError:
            pass

    def step(self):
        self.schedule.step()


class TestNM(TestCase):

    def setUp(self):
        self.environment = NMModel(
            5, 100, 100, 10, 123)

        for i in range(28):
            print(
                i, [agent.location for agent in self.environment.agent],
                )
            self.environment.step()

    def test_total_debris_reach_obstacles(self):
        grid = self.environment.grid
        for obstacle in self.environment.obstacles:
            neighbours = grid.get_neighborhood(
                obstacle.location, obstacle.radius)
            debris_objects = grid.get_objects_from_list_of_grid(
                'Debris', neighbours)
        self.assertEqual(len(debris_objects), 5)
