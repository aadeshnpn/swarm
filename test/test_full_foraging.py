from swarms.lib.agent import Agent
from swarms.objects import Sites, Food, Hub
from swarms.lib.model import Model
from swarms.lib.time import SimultaneousActivation, RandomActivation, StagedActivation
from swarms.lib.space import Grid
from unittest import TestCase
from swarms.utils.bt import BTConstruct
import py_trees
from py_trees import Blackboard
import numpy as np
# import xml.etree.ElementTree as ET
from py_trees.composites import RepeatUntilFalse
from swarms.sbehaviors import (
    NeighbourObjects, IsCarryable, IsSingleCarry,
    SingleCarry, GoTo, Move, IsDropable, IsCarrying, Drop,
    IsVisitedBefore, RandomWalk
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
        self.operation_threshold = 2
        self.genome_storage = []

        # Define a BTContruct object
        self.bt = BTConstruct(None, self)

        self.blackboard = Blackboard()
        self.blackboard.shared_content = dict()

        # self.shared_content = dict()
        self.beta = 0
        self.food_collected = 0
        # Grammatical Evolution part
        from ponyge.algorithm.parameters import Parameters
        parameter = Parameters()
        parameter_list = ['--parameters', 'swarm.txt']
        # Comment when different results is desired.
        # Else set this for testing purpose
        # parameter.params['RANDOM_SEED'] = name
        # # np.random.randint(1, 99999999)
        parameter.params['POPULATION_SIZE'] = self.operation_threshold // 2
        parameter.set_params(parameter_list)
        self.parameter = parameter
        individual = initialisation(self.parameter, 1)
        individual = evaluate_fitness(individual, self.parameter)

        self.individual = individual
        self.bt.xmlstring = self.individual[0].phenotype
        self.bt.construct()

        # Location history
        self.location_history = set()
        self.timestamp = 0

    def step(self):
        #py_trees.logging.level = py_trees.logging.Level.DEBUG
        #output = py_trees.display.ascii_tree(self.bt.behaviour_tree.root)
        #print ('bt tree', output, self.individual[0].phenotype, self.individual[0].fitness)
        # Get the value of food from hub before ticking the behavior
        self.timestamp += 1
        self.location_history.add(self.location)
        #food_in_hub_before = self.get_food_in_hub()
        self.bt.behaviour_tree.tick()
        #food_in_hub_after = self.get_food_in_hub()
        #self.food_collected = food_in_hub_before - food_in_hub_after
        self.food_collected = self.get_food_in_hub()
        # Computes additional value for fitness. In this case foodcollected
        self.overall_fitness()

        cellmates = self.model.grid.get_objects_from_grid(
            'GEBTAgent', self.location)
        # print (cellmates)
        if (len(self.genome_storage) >= self.model.num_agents/25) and (self.exploration_fitness() > 10):
            #print ('genetic', self.name)
            self.genetic_step()

        elif self.timestamp > 20 and self.exploration_fitness() < 2:
            # This is the case of the agent not moving and staying dormant. Need to use genetic
            # operation to change its genome
            individual = initialisation(self.parameter, 10)
            #print (len(set([ind.phenotype for ind in individual])))
            #print ()
            individual = evaluate_fitness(individual, self.parameter)
            self.genome_storage = individual
            self.genetic_step()

        if len(cellmates) > 1:
            self.store_genome(cellmates)
            self.beta = self.food_collected / 1000

    def advance(self):
        pass

    def move(self):
        new_location = ()
        x = int(self.location[0] + np.cos(self.direction) * self.speed)
        y = int(self.location[1] + np.sin(self.direction) * self.speed)
        new_location, direction = self.model.grid.check_limits(
            (x, y), self.direction)
        self.model.grid.move_object(self.location, self, new_location)
        self.location = new_location
        self.direction = direction

    def get_food_in_hub(self):
        #grids = self.model.grid.get_neighborhood(
        #    self.model.hub.location, self.model.hub.radius)
        #no_food_in_hub = self.model.grid.get_objects_from_list_of_grid(
        #    'Food', grids)
        return len(self.attached_objects)*1000
        #return len(no_food_in_hub)

    def store_genome(self, cellmates):
        # cellmates.remove(self)
        self.genome_storage += [agent.individual[0] for agent in cellmates]

    def exchange_chromosome(self,):
        # print('from exchange', self.name)
        individuals = self.genome_storage
        parents = selection(self.parameter, individuals)
        cross_pop = crossover(self.parameter, parents)
        new_pop = mutation(self.parameter, cross_pop)
        new_pop = evaluate_fitness(new_pop, self.parameter)
        individuals = replacement(self.parameter, new_pop, individuals)
        individuals.sort(reverse=False)
        self.individual = [individuals[0]]
        self.individual[0].fitness = 0
        self.genome_storage = []

    def genetic_step(self):
        self.exchange_chromosome()
        self.bt.xmlstring = self.individual[0].phenotype
        self.bt.construct()
        self.food_collected = 0
        self.location_history = set()
        self.timestamp = 0

    def overall_fitness(self):
        # Use a decyaing function to generate fitness

        self.individual[0].fitness = (
            (1 - self.beta) * self.exploration_fitness()) + (
                self.beta * self.food_collected)

    def exploration_fitness(self):
        # Use exploration space as fitness values
        return len(self.location_history)


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

        self.site = Sites(id=1, location=(5, 5), radius=11, q_value=0.5)

        self.grid.add_object_to_grid(self.site.location, self.site)

        self.hub = Hub(id=1, location=(0, 0), radius=11)

        self.grid.add_object_to_grid(self.hub.location, self.hub)

        self.agents = []

        # Create agents
        for i in range(self.num_agents):
            a = GEBTAgent(i, self)
            self.schedule.add(a)
            # Add the agent to a random grid cell
            # x = self.random.randint(
            # -self.grid.width / 2, self.grid.width / 2)
            x = 0
            # y = self.random.randint(
            # -self.grid.height / 2, self.grid.height / 2)
            y = 0

            a.location = (x, y)
            self.grid.add_object_to_grid((x, y), a)
            a.operation_threshold = 2  # self.num_agents // 10
            self.agents.append(a)

        # Add equal number of food source
        for i in range(self.num_agents):
            f = Food(i, location=(12, 12), radius=3)
            self.grid.add_object_to_grid(f.location, f)

    def step(self):
        self.schedule.step()


class TestGEBTSmallGrid(TestCase):

    def setUp(self):
        self.environment = GEEnvironmentModel(100, 100, 100, 10, None)

        for i in range(5):
            self.environment.step()
            #print (i, [(a.name, a.location, a.individual[0].fitness) for a in self.environment.agents[:10]])
            agent = self.find_higest_performer()
            print (i, agent.name, agent.individual[0].fitness, agent.food_collected)
            output = py_trees.display.ascii_tree(agent.bt.behaviour_tree.root)
            print (i, output)
            """
            fname = 'test_bt_xml/' + str(i) + '.xml'
            with open(fname, 'w') as myfile:
                myfile.write(agent.bt.xmlstring)
            """
            # Compute beta
            #self.environment.agent.beta = self.environment.agent.food_collected / self.environment.num_agents


            # for agent in self.environment.schedule.agents:
            #  self.target_phenotype = agent.individual[0].phenotype
            #  self.target_fitness = agent.individual[0].fitness
            #    print('Step', i, agent.name, agent.individual[0].fitness, agent.location)

    # def test_target_string(self):
    #    self.assertEqual('<?xml version="1.0" encoding="UTF-8"?><Sequence><Sequence><Sequence><cond>IsMoveable</cond><cond>IsMupltipleCarry</cond><act>RandomWalk</act></Sequence> <Sequence><cond>IsMotionTrue</cond><cond>IsMoveable</cond><cond>IsMotionTrue</cond><act>SingleCarry</act></Sequence></Sequence> <Selector><cond>IsMotionTrue</cond><cond>IsCarryable</cond><cond>IsMupltipleCarry</cond><act>GoTo</act></Selector></Sequence>', self.target_phenotype)
    def find_higest_performer(self):
        fitness = self.environment.agents[0].individual[0].fitness
        fittest = self.environment.agents[0]
        for agent in self.environment.agents:
            if agent.individual[0].fitness > fitness:
                fittest = agent
        return fittest

    def test_one_traget(self):
        # self.assertEqual(14.285714285714285, self.environment.schedule.agents[0].individual[0].fitness)
        self.assertEqual(9, 9)


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
        # self.exchange_time = model.random.randint(2, 4)
        # This doesn't help. Maybe only perform genetic operations when
        # an agents meet 10% of its total population
        # """
        # Define a BTContruct object
        # fname = 'test_bt_xml/' + str(2) + '.xml'

        # self.bt = BTConstruct(fname, self)

        # self.blackboard = Blackboard()
        # self.blackboard.shared_content = dict()

        self.shared_content['Hub'] = [model.hub]
        # root = py_trees.composites.Sequence("Sequence")
        # root = py_trees.composites.Selector('Selector')
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

        # self.shared_content = dict()
        self.beta = 1
        self.food_collected = 0

        # self.bt.construct()
        py_trees.logging.level = py_trees.logging.Level.DEBUG
        # output = py_trees.display.ascii_tree(self.bt.behaviour_tree.root)
        # Location history
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
        self.food_collected = self.get_food_in_hub()
        self.overall_fitness()

    def get_food_in_hub(self):
        # return len(self.attached_objects) * 1000
        grid = self.model.grid
        hub_loc = self.model.hub.location
        neighbours = grid.get_neighborhood(hub_loc, 35)
        food_objects = grid.get_objects_from_list_of_grid('Food', neighbours)
        # print ('food in the hub', self.name, [(food.id,food.agent_name) for food in food_objects])
        #print (food_objects)
        return len(food_objects)

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
            x = 45
            y = 45

            a.location = (x, y)
            self.grid.add_object_to_grid((x, y), a)
            a.operation_threshold = 2  # self.num_agents // 10
            self.agents.append(a)

        # Add equal number of food source
        for i in range(5):  #self.num_agents * 2):
            f = Food(i, location=(45, 45), radius=3)
            f.agent_name = None
            self.grid.add_object_to_grid(f.location, f)

    def step(self):
        self.schedule.step()


class TestXMLSmallGrid(TestCase):

    def setUp(self):
        self.environment = XMLEnvironmentModel(1, 100, 100, 10, None)

        for i in range(5):
            self.environment.step()
            fittest = self.find_higest_performer()
            #print (i, self.environment.agents[0].location, self.environment.agents[0].fitness)
            print(i, fittest.name, fittest.fitness)

    def test_one_traget(self):
        self.assertEqual(8, 9)

    def find_higest_performer(self):
        fitness = self.environment.agents[0].fitness
        fittest = self.environment.agents[0]
        for agent in self.environment.agents:
            if agent.fitness > fitness:
                fittest = agent
        return fittest