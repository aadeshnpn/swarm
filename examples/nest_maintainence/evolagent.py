"""Derived agent class."""

from swarms.lib.agent import Agent
import numpy as np
from swarms.utils.bt import BTConstruct
from ponyge.operators.initialisation import initialisation
from ponyge.fitness.evaluation import evaluate_fitness
from ponyge.operators.crossover import crossover
from ponyge.operators.mutation import mutation
from ponyge.operators.replacement import replacement
from ponyge.operators.selection import selection

import py_trees


class EvolAgent(Agent):
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
        self.debris_collected = 0
        # Grammatical Evolution part
        from ponyge.algorithm.parameters import Parameters
        parameter = Parameters()
        parameter_list = ['--parameters', '../..,' + model.parm]
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
        self.delayed_reward = 0
        # Location history
        self.location_history = set()
        self.timestamp = 0
        self.step_count = 0

        self.fitness_name = True

    def get_debris_transported(self):
        """Return debris that have been cleared from hub."""
        # Not computational efficient method
        """
        cleared_debris = list(
            filter(
                lambda x: point_distance(
                    hub_loc, x.location) >= (
                        x.initial_distance + 5), self.model.debris))
        """
        grid = self.model.grid
        debris_objects = []
        for obstacle in self.model.obstacles:
            neighbours = grid.get_neighborhood(
                obstacle.location, obstacle.radius)
            debris_objects += grid.get_objects_from_list_of_grid(
                'Debris', neighbours)
        debris_objects = list(set(debris_objects))

        # debris_objects = self.model.grid.get_objects_from_list_of_grid(
        #     'Debris', self.model.neighbours)

        agent_debris_objects = []

        for debris in debris_objects:
                if (
                    debris.agent_name == self.name and
                        self.individual[0].phenotype in list(debris.phenotype.values())):
                    agent_debris_objects.append(debris.weight)
        return sum(agent_debris_objects)

    def detect_debris_carrying(self):
        """Detect if the agent is carrying debris."""
        if len(self.attached_objects) > 0:
            print('Derbis carying', self.name, self.attached_objects)
            output = py_trees.display.ascii_tree(self.bt.behaviour_tree.root)
            print(output)

    def store_genome(self, cellmates):
        """Store the genome from neighbours."""
        # cellmates.remove(self)
        # self.genome_storage += [agent.individual[0] for agent in cellmates]
        for agent in cellmates:
            if agent.debris_collected > 0:
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
        self.debris_collected = 0
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
        # debris has been found, the next block will focus on dropping
        # the debris far from the hub
        div = self.model.modes[self.model.fitmode][0]
        exp = self.model.modes[self.model.fitmode][1]
        pros = self.model.modes[self.model.fitmode][2]
        forg = self.model.modes[self.model.fitmode][3]
        self.individual[0].fitness = (
            (1 - self.beta) * self.delayed_reward  * div + self.exploration_fitness() * exp +
            self.carrying_fitness() * pros + self.debris_collected * forg
            )
        # print(self.name, div, exp, pros, forg, self.individual[0].fitness)
        # self.individual[0].fitness = (1 - self.beta) * self.delayed_reward \
        #     + self.exploration_fitness() + self.carrying_fitness() \
        #     + self.debris_collected

    def carrying_fitness(self):
        """Compute carrying fitness.

        This fitness supports the carrying behavior of
        the agents.
        """
        # return len(self.attached_objects) * (self.timestamp)
        return sum([obj.weight for obj in self.attached_objects])

    def exploration_fitness(self):
        """Compute the exploration fitness."""
        # Use exploration space as fitness values
        return len(self.location_history) - 1

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

        self.location_history.add(self.location)

        # Compute the behavior tree
        self.bt.behaviour_tree.tick()

        # Find the no.of debris collected from the BT execution
        self.debris_collected = self.get_debris_transported()

        # Computes overall fitness using Beta function
        self.overall_fitness()

        cellmates = self.model.grid.get_objects_from_grid(
            type(self).__name__, self.location)

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
        storage_threshold = len(
            self.genome_storage) >= (self.model.num_agents / 10)
        if storage_threshold:
            self.genetic_step()
        elif (
            storage_threshold is False and self.timestamp > 50 and
                self.exploration_fitness() < 10):
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
