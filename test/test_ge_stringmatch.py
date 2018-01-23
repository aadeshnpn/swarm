from unittest import TestCase
from lib.agent import Agent
from lib.model import Model
from lib.time import SimultaneousActivation  # RandomActivation, StagedActivation
from lib.space import Grid
from ponyge.operators.initialisation import initialisation
from ponyge.fitness.evaluation import evaluate_fitness
from ponyge.operators.crossover import crossover
from ponyge.operators.mutation import mutation
from ponyge.operators.replacement import replacement
from ponyge.operators.selection import selection

import numpy as np

# Global variables for width and height
width = 100
height = 100


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
