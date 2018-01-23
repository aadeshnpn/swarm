from unittest import TestCase
from lib.agent import Agent
from lib.model import Model
from lib.time import SimultaneousActivation
from lib.space import Grid
import py_trees
from py_trees import Behaviour, Status

import numpy as np

# Global variables for width and height
width = 100
height = 100


class HasMoney(Behaviour):
    def __init__(self, name):
        super(HasMoney, self).__init__(name)

    def setup(self, timeout, agent):
        self.agent = agent

    def initialise(self):
        pass

    def update(self):
        if self.agent.wealth > 0:
            return Status.SUCCESS
        else:
            return Status.FAILURE


class NeighbourCondition(Behaviour):
    def __init__(self, name):
        super(NeighbourCondition, self).__init__(name)

    def setup(self, timeout, agent):
        self.agent = agent

    def initialise(self):
        pass

    def update(self):
        cellmates = self.agent.model.grid.get_objects_from_grid('SwarmAgent', self.agent.location)
        if len(cellmates) > 1:
            return Status.SUCCESS
        else:
            return Status.FAILURE


class ShareMoney(Behaviour):
    def __init__(self, name):
        super(ShareMoney, self).__init__(name)

    def setup(self, timeout, agent):
        self.agent = agent

    def initialise(self):
        pass

    def update(self):
        try:
            cellmates = self.agent.model.grid.get_objects_from_grid('SwarmAgent', self.agent.location)
            others = self.agent.model.random.choice(cellmates)
            others.wealth += 1
            self.agent.wealth -= 1
            return Status.SUCCESS
        except:
            return Status.FAILURE


class Move(Behaviour):
    def __init__(self, name):
        super(Move, self).__init__(name)

    def setup(self, timeout, agent):
        self.agent = agent

    def initialise(self):
        pass

    def update(self):
        try:
            x = int(self.agent.location[0] + np.cos(self.agent.direction) * self.agent.speed)
            y = int(self.agent.location[1] + np.sin(self.agent.direction) * self.agent.speed)
            new_location, direction = self.agent.model.grid.check_limits((x, y), self.agent.direction)
            self.agent.model.grid.move_object(self.agent.location, self.agent, new_location)
            self.agent.location = new_location
            self.agent.direction = direction
            return Status.SUCCESS
        except:
            return Status.FAILURE


class SwarmAgent(Agent):
    """ An minimalistic swarm agent """
    def __init__(self, name, model):
        super().__init__(name, model)
        self.wealth = 1
        self.location = ()

        self.direction = model.random.rand() * (2 * np.pi)
        self.speed = 2
        self.radius = 3

        root = py_trees.composites.Sequence("Sequence")
        low = Move('4')
        low.setup(0, self)        
        higest = HasMoney('1')
        higest.setup(0, self)
        high = NeighbourCondition('2')
        high.setup(0, self)
        med = ShareMoney('3')
        med.setup(0, self)

        root.add_children([low, higest, high, med])
        self.behaviour_tree = py_trees.trees.BehaviourTree(root)
        
        # """
        # This above part should be replaced by Grammatical Evolution.
        # Based on research, use XML file to generate BT. Parse the XML BT
        # To actually get BT python program gm

    def step(self):
        # """
        # Doing this is equivalent of using behavior tree with four classes
        # in this order, Move, HasMoney, NeighbourCondition, ShareMoney
        # self.move()

        self.behaviour_tree.tick()

    def advance(self):
        pass


class WealthEnvironmentModel(Model):
    """ A environemnt to model swarms """
    def __init__(self, N, width, height, grid=10, seed=None):
        if seed is None:
            super(WealthEnvironmentModel, self).__init__(seed=None)
        else:
            super(WealthEnvironmentModel, self).__init__(seed)

        self.num_agents = N

        self.grid = Grid(width, height, grid)

        self.schedule = SimultaneousActivation(self)

        for i in range(self.num_agents):
            a = SwarmAgent(i, self)
            self.schedule.add(a)
            # Add the agent to a random grid cell
            x = self.random.randint(-self.grid.width / 2, self.grid.width / 2)
            y = self.random.randint(-self.grid.height / 2, self.grid.height / 2)

            a.location = (x, y)
            self.grid.add_object_to_grid((x, y), a)
            a.operation_threshold = self.num_agents // 10

    def step(self):
        self.schedule.step()


class TestWealthSwarmSmallGrid(TestCase):
    
    def setUp(self):
        self.environment = WealthEnvironmentModel(100, 100, 100, 10, 123)

        for i in range(50):
            self.environment.step()

        self.max_wealth = 0
        self.max_agent = 0
        for agent in self.environment.schedule.agents:
            if agent.wealth > self.max_wealth:
                self.max_wealth = agent.wealth
                self.max_agent = agent.name

    def test_maximum_wealth(self):
        self.assertEqual(self.max_wealth, 6)

    def test_maximum_wealth_agent(self):
        self.assertEqual(self.max_agent, 75)
