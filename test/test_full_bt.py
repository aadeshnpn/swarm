from lib.agent import Agent
from swarms.objects import Sites
from lib.model import Model
from lib.time import SimultaneousActivation
from lib.space import Grid
from unittest import TestCase
from swarm.swarms.utils.bt import BTConstruct
import py_trees
import numpy as np
import xml.etree.ElementTree as ET

from swarms.sbehaviors import (
    IsCarryable, IsSingleCarry, SingleCarry,
    NeighbourObjects, IsMultipleCarry, IsInPartialAttached,
    InitiateMultipleCarry, IsEnoughStrengthToCarry,
    Move, GoTo, IsMotionTrue, RandomWalk, IsMoveable,
    MultipleCarry, Away, Towards, IsMultipleCarry
    )

from ponyge.operators.initialisation import initialisation
from ponyge.fitness.evaluation import evaluate_fitness
from ponyge.operators.crossover import crossover
from ponyge.operators.mutation import mutation
from ponyge.operators.replacement import replacement
from ponyge.operators.selection import selection

# Global variables for width and height
width = 100
height = 100


class GEBTAgent(Agent):
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

        # Define a BTContruct object
        self.mapper = BTConstruct(None, None)

        # Grammatical Evolution part
        from ponyge.algorithm.parameters import Parameters
        parameter = Parameters()
        parameter_list = ['--parameters', 'swarm.txt']
        #parameter.params['RANDOM_SEED'] = 1234 # np.random.randint(1, 99999999)        
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
        # self.move()
        # execute  BT 
        self.mapper.xmlstring = self.individual[0].phenotype
        self.mapper.construct()
        py_trees.logging.level = py_trees.logging.Level.DEBUG
        output = py_trees.display.ascii_tree(self.mapper.behaviour_tree.root)
        print (output)

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
        print('from exchange', self.name)
        individuals = self.genome_storage
        parents = selection(self.parameter, individuals)
        cross_pop = crossover(self.parameter, parents)
        new_pop = mutation(self.parameter, cross_pop)
        new_pop = evaluate_fitness(new_pop, self.parameter)
        individuals = replacement(self.parameter, new_pop, individuals)
        individuals.sort(reverse=False)
        self.individual = [individuals[0]]
        self.genome_storage = []    
    

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
            a = GEBTAgent(i, self)
            self.schedule.add(a)
            # Add the agent to a random grid cell
            x = self.random.randint(-self.grid.width / 2, self.grid.width / 2)
            y = self.random.randint(-self.grid.height / 2, self.grid.height / 2)

            a.location = (x, y)
            self.grid.add_object_to_grid((x, y), a)
            a.operation_threshold = self.num_agents // 10

    def step(self):
        self.schedule.step()        


class TestGEBTSmallGrid(TestCase):
    
    def setUp(self):
        self.environment = GEEnvironmentModel(100, 100, 100, 10, 123)

        for i in range(10):
            self.environment.step()

        for agent in self.environment.schedule.agents:
            #self.target_phenotype = agent.individual[0].phenotype
            #self.target_fitness = agent.individual[0].fitness
            print(agent.name, agent.individual[0].fitness)

    #def test_target_string(self):
    #    self.assertEqual('<?xml version="1.0" encoding="UTF-8"?><Sequence><Sequence><Sequence><cond>IsMoveable</cond><cond>IsMupltipleCarry</cond><act>RandomWalk</act></Sequence> <Sequence><cond>IsMotionTrue</cond><cond>IsMoveable</cond><cond>IsMotionTrue</cond><act>SingleCarry</act></Sequence></Sequence> <Selector><cond>IsMotionTrue</cond><cond>IsCarryable</cond><cond>IsMupltipleCarry</cond><act>GoTo</act></Selector></Sequence>', self.target_phenotype)

    def test_one_traget(self):
        self.assertEqual(42.857142857142854, self.target_fitness)