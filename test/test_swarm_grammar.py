"""Test cases for swarm grammar.

This file contains test cases to validate different properties of
swarm grammar.
"""
from swarms.lib.agent import Agent
from swarms.lib.model import Model
from swarms.lib.time import SimultaneousActivation
from swarms.lib.space import Grid
from unittest import TestCase
from swarms.utils.bt import BTConstruct

from py_trees import Blackboard
import py_trees
import numpy as np

from ponyge.operators.initialisation import initialisation

# Global variables for width and height
width = 100
height = 100


class GEBTAgent(Agent):
    """An minimalistic GE agent."""
    def __init__(self, name, model):
        super().__init__(name, model)
        self.location = ()

        self.direction = model.random.rand() * (2 * np.pi)
        self.speed = 2
        self.radius = 3

        self.operation_threshold = 2
        self.genome_storage = []

        # Define a BTContruct object
        self.bt = BTConstruct(None, self)

        self.blackboard = Blackboard()
        self.blackboard.shared_content = dict()

        # Grammatical Evolution part
        from ponyge.algorithm.parameters import Parameters

        parameter = Parameters()
        parameter_list = ['--parameters', 'swarm.txt']
        parameter.params['POPULATION_SIZE'] = self.operation_threshold // 2
        parameter.params['RANDOM_SEED'] = model.seed
        parameter.set_params(parameter_list)
        self.parameter = parameter
        individual = initialisation(self.parameter, 1)

        self.individual = individual

        self.bt.xmlstring = self.individual[0].phenotype

        self.bt.construct()

        self.output = py_trees.display.ascii_tree(self.bt.behaviour_tree.root)

        # Location history
        self.location_history = set()
        self.timestamp = 0

    def step(self):
        """A single step (sense, act) in the environment."""
        self.bt.behaviour_tree.tick()

        self.individual = initialisation(self.parameter, 1)

        self.bt.xmlstring = self.individual[0].phenotype

        self.bt.construct()

        self.output = py_trees.display.ascii_tree(self.bt.behaviour_tree.root)

    def advance(self):
        pass


class GEEnvironmentModel(Model):
    """ A environemnt to model swarms """
    def __init__(self, N, width, height, grid=10, seed=None):
        if seed is None:
            super(GEEnvironmentModel, self).__init__(seed=None)
        else:
            super(GEEnvironmentModel, self).__init__(seed)

        self.num_agents = N

        self.grid = Grid(width, height, grid)

        self.schedule = SimultaneousActivation(self)

        self.agents = []

        # Create agents
        for i in range(self.num_agents):
            a = GEBTAgent(i, self)
            self.schedule.add(a)
            x = 0
            y = 0
            a.location = (x, y)
            self.grid.add_object_to_grid((x, y), a)
            a.operation_threshold = 2  # self.num_agents // 10
            self.agents.append(a)

    def step(self):
        self.schedule.step()


class TestSwarmGrammar(TestCase):

    def setUp(self):

        self.environment = GEEnvironmentModel(1, 100, 100, 10, 123)

    def test_initial_bt(self):

        for i in range(1):
            btree_list = self.environment.agents[0].output.split('-->')

        self.assertEqual(btree_list[3], ' NeighbourObjects102\n    ')

    def test_first_bt(self):

        for i in range(1):
            self.environment.step()
            btree_list = self.environment.agents[0].output.split('-->')

        self.assertEqual(btree_list[1], ' NeighbourObjects172\n        ')

    def test_tenth_bt(self):

        for i in range(10):
            self.environment.step()
            btree_list = self.environment.agents[0].output.split('-->')

        self.assertEqual(btree_list[5], ' CompositeDrop178\n')