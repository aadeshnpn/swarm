from unittest import TestCase
from swarms.lib.agent import Agent
from swarms.lib.model import Model
from swarms.lib.time import SimultaneousActivation  # RandomActivation, StagedActivation
from swarms.lib.space import Grid

import numpy as np


# Global variables for width and height
width = 100
height = 100


class SwarmAgent(Agent):
    """ An minimalistic swarm agent """
    def __init__(self, name, model):
        super().__init__(name, model)
        self.wealth = 1
        self.location = ()

        self.direction = model.random.rand() * (2 * np.pi)
        self.speed = 2
        self.radius = 3

    def step(self):
        if self.wealth > 0:
            self.give_money()

    def advance(self):
        self.move()

    def move(self):
        new_location = ()
        x = int(self.location[0] + np.cos(self.direction) * self.speed)
        y = int(self.location[1] + np.sin(self.direction) * self.speed)
        new_location, direction = self.model.grid.check_limits((x, y), self.direction)
        self.model.grid.move_object(self.location, self, new_location)
        self.location = new_location
        self.direction = direction

    def give_money(self):
        cellmates = self.model.grid.get_objects_from_grid('SwarmAgent', self.location)

        if len(cellmates) > 1:
            other = self.model.random.choice(cellmates)
            other.wealth += 1
            self.wealth -= 1


class EnvironmentModel(Model):
    """ A environemnt to model swarms """
    def __init__(self, N, width, height, grid=10, seed=None):
        if seed is None:
            super(EnvironmentModel, self).__init__(seed=None)
        else:
            super(EnvironmentModel, self).__init__(seed)

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

    def step(self):
        self.schedule.step()


class TestWealthSwarmSmallGrid(TestCase):

    def setUp(self):
        self.environment = EnvironmentModel(100, width, height, 10, 123)

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
        self.assertEqual(self.max_agent, 17)


class TestWealthSwarmBigGrid(TestCase):

    def setUp(self):
        self.environment = EnvironmentModel(1000, 1600, 800, 10, 123)

        for i in range(50):
            self.environment.step()

        self.max_wealth = 0
        self.max_agent = 0

        for agent in self.environment.schedule.agents:
            if agent.wealth > self.max_wealth:
                self.max_wealth = agent.wealth
                self.max_agent = agent.name

    def test_maximum_wealth(self):
        self.assertEqual(self.max_wealth, 4)

    def test_maximum_wealth_agent(self):
        self.assertEqual(self.max_agent, 9)
