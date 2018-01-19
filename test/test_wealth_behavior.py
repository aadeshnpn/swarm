from unittest import TestCase
from lib.agent import Agent
from lib.model import Model
from lib.time import SimultaneousActivation  # RandomActivation, StagedActivation
from lib.space import Grid
import py_trees
from py_trees import Behaviour, Status
from ponyge.operators.initialisation import initialisation
from ponyge.fitness.evaluation import evaluate_fitness
from ponyge.operators.crossover import crossover
from ponyge.operators.mutation import mutation
from ponyge.operators.replacement import replacement, steady_state
from ponyge.operators.selection import selection

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

    #def terminate(self):
    #    pass


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

    #def terminate(self):
    #    pass


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

    #def terminate(self):
    #    pass


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

    #def terminate(self):
    #    pass


class SwarmAgent(Agent):
    """ An minimalistic swarm agent """
    def __init__(self, name, model):
        super().__init__(name, model)
        self.wealth = 1
        self.location = ()

        self.direction = model.random.rand() * (2 * np.pi)
        self.speed = 2
        self.radius = 3

        # """
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

        # Grammatical Evolution part
        from ponyge.algorithm.parameters import Parameters
        parameter = Parameters()
        # list_params_files = ['string_match.txt', 'regression.txt', 'classification.txt']
        parameter_list = ['--parameters', 'string_match_dist.txt']
        parameter.set_params(parameter_list)
        self.parameter = parameter

        individual = initialisation(self.parameter, self.parameter.params['POPULATION_SIZE'])
        individual = evaluate_fitness(individual, self.parameter)
        self.individual = individual
        self.parameter.stats.get_stats(individual)
        # individuals = parameter.params['SEARCH_LOOP'](parameter)
        # parameter.stats.get_stats(individuals, end=True)
        # get_stats(individuals, end=True)

    def step(self):
        # """
        # Doing this is equivalent of using behavior tree with four classes
        # in this order, Move, HasMoney, NeighbourCondition, ShareMoney
        self.move()
        if self.wealth > 0:
            self.give_money()
        # """
        # self.behaviour_tree.tick()

    def advance(self):
        pass

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
            self.exchange_chromosome(cellmates)

    def exchange_chromosome(self, cellmates):
        # If some agents found near. Exchange chromosomes
        individuals = self.individual
        for mate in cellmates:
            individuals += mate.individual
        parents = selection(self.parameter, individuals)
        cross_pop = crossover(self.parameter, parents)
        new_pop = mutation(self.parameter, cross_pop)
        new_pop = evaluate_fitness(new_pop, self.parameter)
        individuals = replacement(self.parameter, new_pop, individuals)
        individuals.sort(reverse=True)
        self.individual = [individuals[0]]
        # print(self.parameter.individual)
        # print(type(self.parameter.individual))
        # exit()


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
        self.environment = EnvironmentModel(100, 100, 100, 10, 123)

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
        self.assertEqual(self.max_wealth, 5)

    def test_maximum_wealth_agent(self):
        self.assertEqual(self.max_agent, 302)
