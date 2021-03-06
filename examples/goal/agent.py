"""Derived agent class."""


import numpy as np
from swarms.lib.agent import Agent
from swarms.utils.bt import BTConstruct
from swarms.utils.results import Results    # noqa : F401

from ponyge.operators.initialisation import initialisation
from ponyge.fitness.evaluation import evaluate_fitness
from ponyge.operators.crossover import crossover
from ponyge.operators.mutation import mutation
from ponyge.operators.replacement import replacement
from ponyge.operators.selection import selection

import py_trees

from swarms.behaviors.sbehaviors import (
    NeighbourObjects, IsVisitedBefore, IsCarrying
    )

from swarms.behaviors.scbehaviors import (
    MoveTowards, Explore, CompositeSingleCarry,
    CompositeDrop
)


class NMAgent(Agent):
    """An minimalistic nest maintanence swarm agent."""

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
        self.debris_collected = 0
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
            try:
                if (
                    debris.agent_name == self.name and
                        debris.phenotype == self.individual[0].phenotype):
                    agent_debris_objects.append(debris)
            except AttributeError:
                pass
        return agent_debris_objects

    def detect_debris_carrying(self):
        """Detect if the agent is carrying debris."""
        if len(self.attached_objects) > 0:
            print('Derbis carying', self.name, self.attached_objects)
            output = py_trees.display.ascii_tree(self.bt.behaviour_tree.root)
            print(output)

    def carrying_fitness(self):
        """Compute carrying fitness.

        This fitness supports the carrying behavior of
        the agents.
        """
        return sum([obj.weight for obj in self.attached_objects])

    def exploration_fitness(self):
        """Compute the exploration fitness."""
        # Use exploration space as fitness values
        location = len(self.location_history)
        if location == 0:
            return 0
        else:
            return location - 1


class LearningAgent(NMAgent):
    """Simple agent with GE capabilities."""

    def __init__(self, name, model):
        """Initialize the agent."""
        super().__init__(name, model)
        self.delayed_reward = 0
        self.phenotypes = dict()

    def init_evolution_algo(self):
        """Agent's GE algorithm operation defination."""
        # Genetic algorithm parameters
        self.operation_threshold = 50
        self.genome_storage = []

        # Grammatical Evolution part
        from ponyge.algorithm.parameters import Parameters
        parameter = Parameters()
        parameter_list = ['--parameters', '../..,goal.txt']
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
        self.beta = 0.9
        self.diversity_fitness = self.individual[0].fitness
        self.individual[0].fitness = 0
        self.generation = 0

    def construct_bt(self):
        """Construct BT."""
        # Get the phenotype of the genome and store as xmlstring
        self.bt.xmlstring = self.individual[0].phenotype
        # Construct actual BT from xmlstring
        self.bt.construct()

    def store_genome(self, cellmates):
        """Store the genome from neighbours."""
        # cellmates.remove(self)
        self.genome_storage += [agent.individual[0] for agent in cellmates]
        # for agent in cellmates:
        #    if agent.debris_collected > 0:
        #        self.genome_storage += agent.individual
        #    elif len(agent.attached_objects) > 0:
        #        self.genome_storage += agent.individual
        #    elif agent.exploration_fitness() > 10:
        #        self.genome_storage += agent.individual

    def exchange_chromosome(self,):
        """Perform genetic operations."""
        # print('from exchange', self.name)
        individuals = self.genome_storage
        parents = selection(self.parameter, individuals)
        cross_pop = crossover(self.parameter, parents)
        new_pop = mutation(self.parameter, cross_pop)
        new_pop = evaluate_fitness(new_pop, self.parameter)
        individuals = replacement(self.parameter, new_pop, individuals)
        individuals.sort(reverse=True)
        self.individual = [individuals[0]]
        self.individual[0].fitness = 0
        self.genome_storage = []

    def genetic_step(self):
        """Additional procedures called after genecti step."""
        # print(
        #    'fitness: ', self.name, self.step_count, self.timestamp,
        #    self.beta,
        #    self.delayed_reward, self.exploration_fitness(),
        #    self.carrying_fitness(), self.food_collected)

        # self.phenotypes[self.individual[0].phenotype] = (
        #    self.exploration_fitness(), self.carrying_fitness(),
        #    self.food_collected)

        self.delayed_reward = self.individual[0].fitness
        self.exchange_chromosome()
        self.bt.xmlstring = self.individual[0].phenotype
        self.bt.construct()
        self.debris_collected = 0
        self.location_history = set()
        self.timestamp = 0
        self.diversity_fitness = self.individual[0].fitness
        self.generation += 1

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
        self.delayed_reward = round(self.beta * self.delayed_reward, 4)
        self.individual[0].fitness = (
            self.ef + self.cf * 4 + self.debris_collected * 8)
        # self.individual[0].fitness = self.delayed_reward \
        #    + self.exploration_fitness() + self.carrying_fitness() \
        #    + self.debris_collected

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
            try:
                if (
                    debris.agent_name == self.name and
                        debris.phenotype == self.individual[0].phenotype):
                    agent_debris_objects.append(debris.weight)
            except AttributeError:
                pass
        return sum(agent_debris_objects)

    def step(self):
        """Take a step in the simulation."""
        # py_trees.logging.level = py_trees.logging.Level.DEBUG
        # output = py_trees.display.ascii_tree(self.bt.behaviour_tree.root)

        # Couting variables
        self.timestamp += 1
        self.step_count += 1

        # Increase beta
        # self.beta = self.timestamp / self.model.iter

        # Compute the behavior tree
        self.bt.behaviour_tree.tick()

        # Maintain location history
        _, gridval = self.model.grid.find_grid(self.location)
        self.location_history.add(gridval)

        # Find the no.of debris collected from the BT execution
        self.debris_collected = self.get_debris_transported()
        # * self.get_food_in_hub(False)

        # Hash the phenotype with its fitness
        # We need to move this from here to genetic step
        self.cf = self.carrying_fitness()
        self.ef = self.exploration_fitness()

        # Computes overall fitness using Beta function
        self.overall_fitness()
        """
        if self.individual[0].phenotype in self.phenotypes.keys():
            e, c, f = self.phenotypes[self.individual[0].phenotype]
            if f < self.debris_collected:
                f = self.debris_collected
            else:
                if c < cf:
                    c = cf
                else:
                    if e < ef:
                        e = ef

            self.phenotypes[self.individual[0].phenotype] = (e, c, f)
        else:
            if int(cf) == 0 and int(ef) == 0 and int(
                    self.debris_collected) == 0:
                pass
            else:
                self.phenotypes[self.individual[0].phenotype] = (
                    self.exploration_fitness(), self.carrying_fitness(),
                    self.debris_collected)
        """
        self.phenotypes = dict()
        self.phenotypes[self.individual[0].phenotype] = (
            self.individual[0].fitness)

        # Find the nearby agents
        cellmates = self.model.grid.get_objects_from_grid(
            type(self).__name__, self.location)

        # If neighbours found, store the genome
        if len(cellmates) > 1:
            self.store_genome(cellmates)

        # Logic for gentic operations.
        # If the genome storage has enough genomes and agents has done some
        # exploration then compute the genetic step OR
        # 600 time step has passed and the agent has not done anything useful
        # then also perform genetic step
        storage_threshold = len(
            self.genome_storage) >= (self.model.num_agents / 10)
        # New logic to invoke genetic step
        if self.individual[0].fitness <= 0 and self.timestamp > 100:
            individual = initialisation(self.parameter, 10)
            individual = evaluate_fitness(individual, self.parameter)
            self.genome_storage = self.genome_storage + individual
            self.genetic_step()
        elif (
                (
                    self.individual[0].fitness >= 0 and storage_threshold
                    ) and self.timestamp > 200 and self.debris_collected <= 0):
            self.genetic_step()


class ExecutingAgent(NMAgent):
    """A nest maintanance swarm agent.

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
        # py_trees.display.render_dot_tree(
        #    self.bt.behaviour_tree.root, name='/tmp/' + str(self.name))

    def step(self):
        """Agent action at a single time step."""
        # Maintain the location history of the agent
        # self.location_history.add(self.location)

        # Compute the behavior tree
        self.bt.behaviour_tree.tick()

        # Find the no.of food collected from the BT execution
        # self.food_collected = len(self.get_food_in_hub())


class TestingAgent(NMAgent):
    """A nest maintanance swarm agent.

    This agent will run the hand-coded behaviors evolved.
    """

    def __init__(self, name, model, xmlstring=None):
        """Initialize the agent."""
        super().__init__(name, model)
        self.xmlstring = xmlstring

    def construct_bt(self):
        """Construct BT."""
        # Get the phenotype of the genome and store as xmlstring
        # self.bt.xmlstring = self.xmlstring
        # Drop branch
        # Drop branch
        dseq = py_trees.composites.Sequence('DSequence')
        iscarrying = IsCarrying('IsCarrying_Debris')
        iscarrying.setup(0, self, 'Debris')

        neighhub = NeighbourObjects('NeighbourObjects')
        neighhub.setup(0, self, 'Obstacles')

        drop = CompositeDrop('CompositeDrop_Debris')
        drop.setup(0, self, 'Debris')

        dseq.add_children([neighhub, drop])

        # Carry branch
        cseq = py_trees.composites.Sequence('CSequence')

        neighsite = py_trees.meta.inverter(NeighbourObjects)(
            'NeighbourObjects')
        neighsite.setup(0, self, 'Obstacles')

        neighfood = NeighbourObjects('NeighbourObjects_Debris')
        neighfood.setup(0, self, 'Debris')

        invcarrying = py_trees.meta.inverter(IsCarrying)(
            'IsCarrying_Debris')
        invcarrying.setup(0, self, 'Debris')

        carry = CompositeSingleCarry('CompositeSingleCarry_Debris')
        carry.setup(0, self, 'Debris')

        cseq.add_children([neighsite, neighfood, invcarrying, carry])

        # Locomotion branch

        # Move to site
        siteseq = py_trees.composites.Sequence('SiteSeq')

        sitefound = IsVisitedBefore('IsVisitedBefore')
        sitefound.setup(0, self, 'Obstacles')

        gotosite = MoveTowards('MoveTowards')
        gotosite.setup(0, self, 'Obstacles')

        siteseq.add_children([sitefound, iscarrying, gotosite])

        # Move to hub
        hubseq = py_trees.composites.Sequence('HubSeq')

        gotohub = MoveTowards('MoveTowards_Hub')
        gotohub.setup(0, self, 'Hub')

        hubseq.add_children([iscarrying, gotohub])

        sitenotfound = py_trees.meta.inverter(IsVisitedBefore)(
            'IsVisitedBefore')
        sitenotfound.setup(0, self, 'Obstacles')

        explore = Explore('Explore')
        explore.setup(0, self)

        randwalk = py_trees.composites.Sequence('Randwalk')
        randwalk.add_children([explore])

        locoselect = py_trees.composites.Selector('Move')
        locoselect.add_children([siteseq, explore])
        select = py_trees.composites.Selector('Main')

        select.add_children([dseq, cseq, locoselect])

        self.behaviour_tree = py_trees.trees.BehaviourTree(select)
        # Construct actual BT from xmlstring
        # self.bt.construct()

    def step(self):
        """Agent action at a single time step."""
        # Maintain the location history of the agent
        # self.location_history.add(self.location)

        # Compute the behavior tree
        # self.bt.behaviour_tree.tick()
        self.behaviour_tree.tick()

        # Find the no.of food collected from the BT execution
        # self.food_collected = len(self.get_food_in_hub())
