"""Derived agent class."""

from swarms.lib.agent import Agent
import numpy as np
from swarms.utils.bt import BTConstruct
from swarms.utils.results import Results    # noqa : F401

from ponyge.operators.initialisation import initialisation
from ponyge.fitness.evaluation import evaluate_fitness
from ponyge.operators.crossover import crossover
from ponyge.operators.mutation import mutation
from ponyge.operators.replacement import replacement
from ponyge.operators.selection import selection

import py_trees


class ForagingAgent(Agent):
    """An minimalistic foraging swarm agent."""

    def __init__(self, name, model):
        """Initialize the agent."""
        super().__init__(name, model)
        self.location = ()
        # Agent attributes for motion
        self.direction = model.random.rand() * (2 * np.pi)
        self.speed = 2
        self.radius = 3
        self.carryable = False
        self.shared_content = dict()
        self.food_collected = 0
        # Results
        self.results = "db"  # This can take 2 values. db or file
        # Behavior Tree Class
        self.bt = BTConstruct(None, self)
        # Location history
        self.location_history = set()
        # Step count
        self.timestamp = 0
        self.step_count = 0

        self.fitness_name = True

    def init_evolution_algo(self):
        """Agent's GE algorithm operation defination."""
        # This is a abstract class. Only the agents
        # with evolving nature require this
        pass

    def construct_bt(self):
        """Abstract method to construct BT."""
        # Different agents has different way to construct the BT
        pass

    # New Agent methods for behavior based robotics
    def sense(self):
        """Sense included in behavior tree."""
        pass

    def plan(self):
        """Plan not required for now."""
        pass

    def step(self):
        """Agent action at a single time step."""
        # Abstract class only evoling agent need this class
        pass

    def advance(self):
        """Require for staged activation."""
        pass

    def get_food_in_hub(self):
        """Get the food in the hub stored by the agent."""
        grid = self.model.grid
        hub_loc = self.model.hub.location
        neighbours = grid.get_neighborhood(hub_loc, 10)
        food_objects = grid.get_objects_from_list_of_grid('Food', neighbours)
        agent_food_objects = []
        for food in food_objects:
            if food.agent_name == self.name:
                agent_food_objects.append(food)
        return agent_food_objects

    def detect_food_carrying(self):
        """Check if the agent is carrying food."""
        if len(self.attached_objects) > 0:
            print('Food carying', self.name, self.attached_objects)
            output = py_trees.display.ascii_tree(self.bt.behaviour_tree.root)
            print(output)

    def carrying_fitness(self):
        """Compute carrying fitness.

        This fitness supports the carrying behavior of
        the agents.
        """
        return len(self.attached_objects) * (self.timestamp)

    def exploration_fitness(self):
        """Compute the exploration fitness."""
        # Use exploration space as fitness values
        return len(self.location_history) - 1


class LearningAgent(ForagingAgent):
    """Simple agent with GE capabilities."""

    def __init__(self, name, model):
        """Initialize the agent."""
        super().__init__(name, model)
        self.delayed_reward = 0

    def init_evolution_algo(self):
        """Agent's GE algorithm operation defination."""
        # Genetic algorithm parameters
        self.operation_threshold = 2
        self.genome_storage = []

        # Grammatical Evolution part
        from ponyge.algorithm.parameters import Parameters
        parameter = Parameters()
        parameter_list = ['--parameters', '../..,swarm.txt']
        # Comment when different results is desired.
        # Else set this for testing purpose
        # parameter.params['RANDOM_SEED'] = name
        # # np.random.randint(1, 99999999)
        # Set GE runtime parameters
        parameter.params['POPULATION_SIZE'] = self.operation_threshold // 2
        parameter.set_params(parameter_list)
        self.parameter = parameter
        # Initialize the genome
        individual = initialisation(self.parameter, 1)
        individual = evaluate_fitness(individual, self.parameter)
        # Assign the genome to the agent
        self.individual = individual
        # Fitness
        self.beta = 0.0001
        self.diversity_fitness = self.individual[0].fitness

    def construct_bt(self):
        """Construct BT."""
        # Get the phenotype of the genome and store as xmlstring
        self.bt.xmlstring = self.individual[0].phenotype
        # Construct actual BT from xmlstring
        self.bt.construct()

    def store_genome(self, cellmates):
        """Store the genome from neighbours."""
        # cellmates.remove(self)
        # self.genome_storage += [agent.individual[0] for agent in cellmates]
        for agent in cellmates:
            if agent.food_collected > 0:
                self.genome_storage += agent.individual
            elif len(agent.attached_objects) > 0:
                self.genome_storage += agent.individual
            elif agent.exploration_fitness() > 10:
                self.genome_storage += agent.individual

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
        self.delayed_reward = self.individual[0].fitness
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
        self.individual[0].fitness = (1 - self.beta) * self.delayed_reward \
            + self.exploration_fitness() + self.carrying_fitness() \
            + self.food_collected

    def step(self):
        """Take a step in the simulation."""
        # py_trees.logging.level = py_trees.logging.Level.DEBUG
        # output = py_trees.display.ascii_tree(self.bt.behaviour_tree.root)

        # Couting variables
        self.timestamp += 1
        self.step_count += 1

        # Increase beta
        self.beta = self.step_count / self.model.iter

        # Maintain location history
        self.location_history.add(self.location)

        # Compute the behavior tree
        self.bt.behaviour_tree.tick()

        # Find the no.of food collected from the BT execution
        self.food_collected = len(self.get_food_in_hub())

        # Computes overall fitness using Beta function
        self.overall_fitness()
        # Find the nearby agents
        cellmates = self.model.grid.get_objects_from_grid(
            type(self).__name__, self.location)
        # Logic for gentic operations.
        # If the genome storage has enough genomes and agents has done some
        # exploration then compute the genetic step OR
        # 600 time step has passed and the agent has not done anything useful
        # then also perform genetic step
        storage_threshold = len(
            self.genome_storage) >= (self.model.num_agents / 4)
        if storage_threshold:
            self.genetic_step()
        elif (
                (
                    storage_threshold is False and self.timestamp > 50
                    ) and (self.exploration_fitness() < 10)):
            individual = initialisation(self.parameter, 10)
            individual = evaluate_fitness(individual, self.parameter)
            self.genome_storage = individual
            self.genetic_step()

        # If neighbours found, store the genome
        if len(cellmates) > 1:
            self.store_genome(cellmates)


class ExecutingAgent(ForagingAgent):
    """A foraging swarm agent.

    This agent will run the behaviors evolved.
    """

    def __init__(self, name, model, xmlstring=None):
        """Initialize the agent."""
        super().__init__(name, model)
        self.xmlstring = xmlstring

    def construct_bt(self):
        """Construct BT."""
        # Get the phenotype of the genome and store as xmlstring
        self.bt.xmlstring = self.xmlstring
        # Construct actual BT from xmlstring
        self.bt.construct()

    def step(self):
        """Agent action at a single time step."""
        # Maintain the location history of the agent
        # self.location_history.add(self.location)

        # Compute the behavior tree
        self.bt.behaviour_tree.tick()

        # Find the no.of food collected from the BT execution
        # self.food_collected = len(self.get_food_in_hub())
