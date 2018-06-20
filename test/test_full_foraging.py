from swarms.lib.agent import Agent
from swarms.objects import Sites, Food, Hub
from swarms.lib.model import Model
from swarms.lib.time import SimultaneousActivation
from swarms.lib.space import Grid
from unittest import TestCase
import py_trees

import numpy as np
# import xml.etree.ElementTree as ET
from py_trees.composites import RepeatUntilFalse
from swarms.sbehaviors import (
    NeighbourObjects, IsCarryable, IsSingleCarry,
    SingleCarry, GoTo, Move, IsDropable, IsCarrying, Drop,
    IsVisitedBefore, RandomWalk
)

# Global variables for width and height
width = 100
height = 100


class XMLTestAgent(Agent):
    """ An minimalistic GE agent """
    def __init__(self, name, model):
        super().__init__(name, model)
        self.location = ()

        self.direction = model.random.rand() * (2 * np.pi)
        self.speed = 2
        self.radius = 3

        self.moveable = True
        self.shared_content = dict()

        self.shared_content['Hub'] = {model.hub}
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

        self.beta = 1
        self.food_collected = 0

        self.location_history = set()
        self.timestamp = 0

    def advance(self):
        pass

    def step(self):
        self.timestamp += 1
        self.location_history.add(self.location)
        self.behaviour_tree.tick()
        # self.blackboard = Blackboard()
        # self.blackboard.shared_content = dict()
        self.food_collected = len(self.get_food_in_hub())
        self.overall_fitness()

    def get_food_in_hub(self):
        # return len(self.attached_objects) * 1000
        grid = self.model.grid
        hub_loc = self.model.hub.location
        neighbours = grid.get_neighborhood(hub_loc, 35)
        food_objects = grid.get_objects_from_list_of_grid('Food', neighbours)

        # print (food_objects)
        return food_objects

    def overall_fitness(self):
        # Use a decyaing function to generate fitness

        self.fitness = (
            (1 - self.beta) * self.exploration_fitness()) + (
                self.beta * self.food_collected)

    def exploration_fitness(self):
        # Use exploration space as fitness values
        return len(self.location_history)


class XMLEnvironmentModel(Model):
    """ A environemnt to model swarms """
    def __init__(self, N, width, height, grid=10, seed=None):
        if seed is None:
            super(XMLEnvironmentModel, self).__init__(seed=None)
        else:
            super(XMLEnvironmentModel, self).__init__(seed)

        self.num_agents = N

        self.grid = Grid(width, height, grid)

        self.schedule = SimultaneousActivation(self)

        self.site = Sites(id=1, location=(45, 45), radius=5, q_value=0.5)

        self.grid.add_object_to_grid(self.site.location, self.site)

        self.hub = Hub(id=1, location=(0, 0), radius=11)

        self.grid.add_object_to_grid(self.hub.location, self.hub)

        self.agents = []

        # Create agents
        for i in range(self.num_agents):
            a = XMLTestAgent(i, self)
            self.schedule.add(a)
            x = 25
            y = 25

            a.location = (x, y)
            self.grid.add_object_to_grid((x, y), a)
            a.operation_threshold = 2  # self.num_agents // 10
            self.agents.append(a)

        # Add equal number of food source
        for i in range(5):  # self.num_agents * 2):
            f = Food(i, location=(45, 45), radius=3)
            f.agent_name = None
            self.grid.add_object_to_grid(f.location, f)

    def step(self):
        self.schedule.step()


class TestXMLSmallGrid(TestCase):

    def setUp(self):
        self.environment = XMLEnvironmentModel(50, 100, 100, 10, 102)

        for i in range(20):
            self.environment.step()
            self.food_objects = self.environment.agents[0].get_food_in_hub()
            print('food in the hub', i, [(
                food.id, food.agent_name) for food in self.food_objects])

    def test_one_traget(self):
        self.assertEqual(5, len(self.food_objects))
