"""Derived agent class."""

from swarms.lib.agent import Agent
import numpy as np
from swarms.utils.bt import BTConstruct
from swarms.utils.results import Results

from ponyge.operators.initialisation import initialisation
from ponyge.fitness.evaluation import evaluate_fitness
from ponyge.operators.crossover import crossover
from ponyge.operators.mutation import mutation
from ponyge.operators.replacement import replacement
from ponyge.operators.selection import selection

import py_trees


class SwarmAgent(Agent):
    """An minimalistic swarm agent."""

    def __init__(self, name, model):
        """Initialize the agent."""
        super().__init__(name, model)
        self.location = ()

        self.direction = model.random.rand() * (2 * np.pi)
        self.speed = 2
        self.radius = 3
        self.results = "db"  # This can take 2 values. db or file

        # self.exchange_time = model.random.randint(2, 4)
        # This doesn't help. Maybe only perform genetic operations when
        # an agents meet 10% of its total population
        # """
        self.operation_threshold = 2
        self.genome_storage = []

        # Define a BTContruct object
        self.bt = BTConstruct(None, self)

        # self.blackboard = Blackboard()
        # self.blackboard.shared_content = dict()

        self.shared_content = dict()
        # self.shared_content = dict(
        self.carryable = False
        self.beta = 0.0001
        self.food_collected = 0
        # Grammatical Evolution part
        from ponyge.algorithm.parameters import Parameters
        parameter = Parameters()
        parameter_list = ['--parameters', '../..,swarm.txt']
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

        self.diversity_fitness = self.individual[0].fitness

        # Location history
        self.location_history = set()
        self.timestamp = 0
        self.step_count = 0

        self.fitness_name = True

    def get_food_in_hub(self):
        # return len(self.attached_objects) * 1000
        grid = self.model.grid
        hub_loc = self.model.hub.location
        neighbours = grid.get_neighborhood(hub_loc, 10)
        food_objects = grid.get_objects_from_list_of_grid('Food', neighbours)
        agent_food_objects = []
        for food in food_objects:
            if food.agent_name == self.name:
                agent_food_objects.append(food)
        # print (food_objects)
        return agent_food_objects

    def detect_food_carrying(self):
        if len(self.attached_objects) > 0:
            print('Food carying', self.name, self.attached_objects)
            output = py_trees.display.ascii_tree(self.bt.behaviour_tree.root)
            print(output)

    def store_genome(self, cellmates):
        """Store the genome from neighbours."""
        # cellmates.remove(self)
        self.genome_storage += [agent.individual[0] for agent in cellmates]

    def exchange_chromosome(self,):
        """Perform genetic operations."""
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
        """Additional procedures called after genecti step."""
        self.exchange_chromosome()
        self.bt.xmlstring = self.individual[0].phenotype
        self.bt.construct()
        self.food_collected = 0
        self.location_history = set()
        self.timestamp = 0
        self.diversity_fitness = self.individual[0].fitness

    def overall_fitness(self):
        """Compute complete fitness.

        Goals are represented by objective function. We use combination of
        objective function to define overall fitness of the agents
        performance.
        """
        # Use a decyaing function to generate fitness
        # Use two step decaying function
        # First block gives importance to exploration and when as soon
        # food has been found, the next block will focus on dropping
        # the food on hub
        """
        if self.carrying_fitness() <= 0 and self.food_collected <= 0:
            self.individual[0].fitness = (
                (1 - self.beta) * self.exploration_fitness(
                ) + self.beta * self.carrying_fitness())
            self.fitness_name = True

        elif self.carrying_fitness() > 0:
            self.individual[0].fitness = (
                1 - self.beta) * self.carrying_fitness() + (
                    self.beta * self.food_collected * self.timestamp)
            self.fitness_name = False
        """
        if 'Sites' not in self.shared_content.keys():
            self.individual[0].fitness = self.exploration_fitness()
            self.fitness_name = 'EXP'
        elif 'Sites' in self.shared_content.keys():
            self.individual[0].fitness = (
                (1 - self.beta) * self.exploration_fitness(
                ) + self.beta * self.carrying_fitness())
            self.fitness_name = 'CAR'
        elif 'Sites' in self.shared_content.keys() and self.carrying_fitness() > 0:
            self.individual[0].fitness = (self.beta * self.food_collected * self.timestamp)
            self.fitness_name = 'DRP'

    def carrying_fitness(self):
        """Compute carrying fitness.

        This fitness supports the carrying behavior of
        the agents.
        """
        return len(self.attached_objects) * (1.0 / self.timestamp)

    def exploration_fitness(self):
        """Compute the exploration fitness."""
        # Use exploration space as fitness values
        return len(self.location_history)

    # New Agent methods for behavior based robotics
    def sense(self):
        """Sense included in behavior tree."""
        pass

    def plan(self):
        """Plan not required for now."""
        pass

    def step(self):
        """Agent action at a single time step."""
        # py_trees.logging.level = py_trees.logging.Level.DEBUG
        # output = py_trees.display.ascii_tree(self.bt.behaviour_tree.root)

        # Couting variables
        self.timestamp += 1
        self.step_count += 1

        # Increase beta
        self.beta = self.step_count / self.model.iter

        # Maintain the location history of the agent
        self.location_history.add(self.location)

        # Compute the behavior tree
        self.bt.behaviour_tree.tick()

        # Find the no.of food collected from the BT execution
        self.food_collected = len(self.get_food_in_hub())

        # Computes overall fitness using Beta function
        self.overall_fitness()
        if self.fitness_name != 'EXP':
            print (self.step_count, self.name, np.round(self.individual[0].fitness,2),
            self.exploration_fitness(), self.timestamp, np.round(self.beta,2), self.fitness_name, self.shared_content, )
        # Find the nearby agents
        cellmates = self.model.grid.get_objects_from_grid(
            'SwarmAgent', self.location)

        # Create a results instance and save it to a file
        """
        self.results = Results(
            self.model.pname, self.model.connect, self.model.sn, self.name,
            self.step_count, self.timestamp, self.beta,
            self.individual[0].fitness,
            self.diversity_fitness, self.exploration_fitness(),
            self.food_collected, len(cellmates), self.individual[0].genome,
            self.individual[0].phenotype, self.bt
            )
        """
        # Save the results to a db
        # self.results.save_to_file()

        # Logic for gentic operations.
        # If the genome storage has enough genomes and agents has done some
        # exploration then compute the genetic step OR
        # 600 time step has passed and the agent has not done anything useful
        # then also perform genetic step
        if (len(self.genome_storage) >= self.model.num_agents / 1.4) \
                and (self.exploration_fitness() >= 10):
                    self.genetic_step()
        elif (self.timestamp > 100 and self.exploration_fitness() < 10):
            # This is the case of the agent not moving and staying dormant.
            # Need to use genetic operation to change its genome
            individual = initialisation(self.parameter, 10)
            individual = evaluate_fitness(individual, self.parameter)
            self.genome_storage = individual
            self.genetic_step()

        # If neighbours found, store the genome
        if len(cellmates) > 1:
            self.store_genome(cellmates)

    def advance(self):
        """Require for staged activation."""
        pass


class RunSwarmAgent(Agent):
    """A swarm agent.

    This agent will run the behaviors evolved.
    """

    def __init__(self, name, model, xmlstring=None):
        """Initialize the agent."""
        super().__init__(name, model)
        self.location = ()

        self.direction = model.random.rand() * (2 * np.pi)
        self.speed = 2
        self.radius = 3
        self.carryable = False

        # Define a BTContruct object
        self.bt = BTConstruct(None, self)

        self.shared_content = dict()

        self.food_collected = 0

        self.bt.xmlstring = xmlstring
        self.bt.construct()

        # self.diversity_fitness = model.fitness

        # Location history
        self.location_history = set()
        self.timestamp = 0
        self.step_count = 0

    def get_food_in_hub(self):
        # return len(self.attached_objects) * 1000
        grid = self.model.grid
        hub_loc = self.model.hub.location
        neighbours = grid.get_neighborhood(hub_loc, 10)
        food_objects = grid.get_objects_from_list_of_grid('Food', neighbours)
        agent_food_objects = []
        for food in food_objects:
            if food.agent_name == self.name:
                agent_food_objects.append(food)
        # print (food_objects)
        return agent_food_objects

    def detect_food_carrying(self):
        if len(self.attached_objects) > 0:
            print('Food carying', self.name, self.attached_objects)
            output = py_trees.display.ascii_tree(self.bt.behaviour_tree.root)
            print(output)

    def carrying_fitness(self):
        """Compute carrying fitness.

        This fitness supports the carrying behavior of
        the agents.
        """
        return len(self.attached_objects)

    def exploration_fitness(self):
        """Compute the exploration fitness."""
        # Use exploration space as fitness values
        return len(self.location_history)

    # New Agent methods for behavior based robotics
    def sense(self):
        """Sense included in behavior tree."""
        pass

    def plan(self):
        """Plan not required for now."""
        pass

    def step(self):
        """Agent action at a single time step."""

        # Maintain the location history of the agent
        # self.location_history.add(self.location)

        # Compute the behavior tree
        self.bt.behaviour_tree.tick()

        # Find the no.of food collected from the BT execution
        # self.food_collected = len(self.get_food_in_hub())

    def advance(self):
        """Require for staged activation."""
        pass
