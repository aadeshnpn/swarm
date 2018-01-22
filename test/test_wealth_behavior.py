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


class GEAgent(Agent):
    """ An minimalistic GE agent """
    def __init__(self, name, model):
        super().__init__(name, model)
        self.location = ()

        self.direction = model.random.rand() * (2 * np.pi)
        self.speed = 2
        self.radius = 3
        
        # self.exchange_time = model.random.randint(2, 4)
        # This doesn't help. Maybe only perform genetic operations when 
        # an agents meet 10% of its total population
        # """
        self.operation_threshold = 10
        self.genome_storage = []

        # Grammatical Evolution part
        from ponyge.algorithm.parameters import Parameters
        parameter = Parameters()
        # list_params_files = ['string_match.txt', 'regression.txt', 'classification.txt']
        parameter_list = ['--parameters', 'string_match_dist.txt']
        parameter.params['RANDOM_SEED'] = 1234 #np.random.randint(1, 99999999)        
        parameter.params['POPULATION_SIZE'] = self.operation_threshold // 2         
        parameter.set_params(parameter_list)
        self.parameter = parameter
        individual = initialisation(self.parameter, 1)
        individual = evaluate_fitness(individual, self.parameter)
        self.individual = individual

    def step(self):
        # """
        # Doing this is equivalent of using behavior tree with four classes
        # in this order, Move, HasMoney, NeighbourCondition, ShareMoney
        self.move()

        cellmates = self.model.grid.get_objects_from_grid('GEAgent', self.location)

        if len(self.genome_storage) >= self.operation_threshold:
            self.exchange_chromosome(cellmates)

        if len(cellmates) > 1:
            self.store_genome(cellmates)

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

    def store_genome(self, cellmates):
        # cellmates.remove(self)
        self.genome_storage += [agent.individual[0] for agent in cellmates]

    def exchange_chromosome(self, cellmates):
        individuals = self.genome_storage
        parents = selection(self.parameter, individuals)
        cross_pop = crossover(self.parameter, parents)
        new_pop = mutation(self.parameter, cross_pop)
        new_pop = evaluate_fitness(new_pop, self.parameter)
        individuals = replacement(self.parameter, new_pop, individuals)
        individuals.sort(reverse=True)
        self.individual = [individuals[0]]
        self.genome_storage = []    
    
class TestGESmallGrid(TestCase):
    
    def setUp(self):
        self.environment = GEEnvironmentModel(100, 100, 100, 10, 123)

        for i in range(200):
            self.environment.step()

        self.one_target = False
        for agent in self.environment.schedule.agents:
            self.target = agent.individual[0].phenotype

            if agent.individual[0].phenotype == 'Hello':
                self.one_target = True

    def test_target_string(self):
        self.assertEqual(self.target, 'Hello')

    def test_one_traget(self):
        self.assertEqual(self.one_target, True)


class SwarmAgent(Agent):
    """ An minimalistic swarm agent """
    def __init__(self, name, model):
        super().__init__(name, model)
        self.wealth = 1
        self.location = ()

        self.direction = model.random.rand() * (2 * np.pi)
        self.speed = 2
        self.radius = 3
        
        # self.exchange_time = model.random.randint(2, 4)
        # This doesn't help. Maybe only perform genetic operations when 
        # an agents meet 10% of its total population
        # """
        self.operation_threshold = 10 #model.random.randint(4, 10)
        self.genome_storage = []
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
        self.move()

        cellmates = self.model.grid.get_objects_from_grid('SwarmAgent', self.location)

        if self.wealth > 0:
            self.give_money(cellmates)
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

    def give_money(self, cellmates):
        if len(cellmates) > 1:
            other = self.model.random.choice(cellmates)
            other.wealth += 1
            self.wealth -= 1


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

        for i in range(self.num_agents):
            a = GEAgent(i, self)
            self.schedule.add(a)
            # Add the agent to a random grid cell
            x = self.random.randint(-self.grid.width / 2, self.grid.width / 2)
            y = self.random.randint(-self.grid.height / 2, self.grid.height / 2)

            a.location = (x, y)
            self.grid.add_object_to_grid((x, y), a)
            a.operation_threshold = self.num_agents // 10
        # exit()

    def step(self):
        self.schedule.step()


class TestWealthSwarmSmallGrid(TestCase):
    
    def setUp(self):
        self.environment = WealthEnvironmentModel(100, 100, 100, 10, 123)

        for i in range(50):
            self.environment.step()

        self.max_wealth = 0
        self.max_agent = 0
        self.one_target = False
        for agent in self.environment.schedule.agents:
            if agent.wealth > self.max_wealth:
                self.max_wealth = agent.wealth
                self.max_agent = agent.name

    def test_maximum_wealth(self):
        self.assertEqual(self.max_wealth, 6)

    def test_maximum_wealth_agent(self):
        self.assertEqual(self.max_agent, 75)

"""
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
"""