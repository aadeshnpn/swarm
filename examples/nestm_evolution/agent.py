from inspect import CO_ITERABLE_COROUTINE
from py_trees import common
from py_trees.composites import Selector, Parallel
from py_trees import common, blackboard
from py_trees.trees import BehaviourTree
from swarms.behaviors.sbehaviors import DropCue, SendSignal, ObjectsStore
import numpy as np
from swarms.lib.agent import Agent
from swarms.utils.bt import BTConstruct
from swarms.utils.results import Results    # noqa : F401
from swarms.utils.distangle import point_distance

from ponyge.operators.initialisation import initialisation
from ponyge.fitness.evaluation import evaluate_fitness
from ponyge.operators.crossover import crossover
from ponyge.operators.mutation import mutation
from ponyge.operators.replacement import replacement
from ponyge.operators.selection import selection

import py_trees
import copy
from flloat.parser.ltlf import LTLfParser


class NestAgent(Agent):
    """An minimalistic nest maintenance swarm agent."""

    def __init__(self, name, model):
        """Initialize the agent."""
        super().__init__(name, model)
        self.location = ()
        self.prev_location = ()
        # Agent attributes for motion
        self.direction = model.random.rand() * (2 * np.pi)
        self.speed = 2
        self.radius = 3
        self.shared_content = dict()
        self.food_collected = 0
        # Results
        self.results = "db"  # This can take 2 values. db or file
        # Behavior Tree Class
        self.bt = BTConstruct(None, self)
        # Location history
        self.location_history = set()

        # Communication history
        self.signal_time = 0
        self.no_cue_dropped = 0
        # Step count
        self.timestamp = 0
        self.step_count = 0

        self.fitness_name = True
        self.ltrate = 0
        self.geneticrate = False

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

    def get_debris_transported(self, distance_threshold=35):
        """Return debris that have been cleared from hub."""
        # Not computational efficient method
        debris_objects = []
        for debry in self.model.debris:
            distance = point_distance(debry.location, self.model.hub.location)
            if distance > distance_threshold:
                debris_objects.append(debry)

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
        # return len(self.attached_objects) * (self.timestamp)
        return sum([obj.weight for obj in self.attached_objects])

    def exploration_fitness(self):
        """Compute the exploration fitness."""
        # Use exploration space as fitness values
        locations = len(self.location_history)
        if locations == 0:
            return 0
        else:
            return locations - 1

    def communication_fitness(self):
        """Compute communication fitness. """

        childrens = list(self.bt.behaviour_tree.root.iterate())
        for child in childrens:
            if isinstance(child, SendSignal) and child.status == py_trees.Status.SUCCESS:
                self.signal_time += 1
            if isinstance(child, DropCue) and child.status == py_trees.Status.SUCCESS:
                self.no_cue_dropped += 1
        return self.signal_time + self.no_cue_dropped

    """Function related to LTLf and goals."""
    def evaluate_trace(self, goalspec, trace):
        # Evaluate the trace
        parser = LTLfParser()
        parsed_formula = parser(goalspec)
        result = parsed_formula.truth(trace)
        return result


class LearningAgent(NestAgent):
    """Simple agent with GE capabilities."""

    def __init__(self, name, model, threshold=5):
        """Initialize the agent."""
        super().__init__(name, model)
        self.delayed_reward = 0
        self.phenotypes = dict()

        # Flags to make the computation easier
        self.avoid_trap = False
        self.avoid_obs = False
        self.blackboard = blackboard.Client(name=str(self.name))
        self.blackboard.register_key(key='neighbourobj', access=common.Access.WRITE)
        self.blackboard.neighbourobj = dict()

        self.selectors_reward = 0
        self.constraints_reward = 0
        self.postcond_reward = 0
        self.threshold = threshold

    def evaluate_constraints_conditions(self):
        allnodes = list(self.bt.behaviour_tree.root.iterate())
        selectors = list(filter(
            lambda x: isinstance(x, Selector), allnodes)
            )

        constraints = list(filter(
            lambda x: x.name.split('_')[-1] == 'constraint', allnodes)
            )

        postcond = list(filter(
            lambda x: x.name.split('_')[-1] == 'postcond', allnodes)
            )
        # print([(node.name) for node in allnodes])
        self.selectors_reward = sum([1 for sel in selectors if sel.status == common.Status.SUCCESS])
        self.constraints_reward = sum([-2 for const in constraints if const.status == common.Status.FAILURE])
        self.postcond_reward = sum([1 for pcond in postcond if pcond.status == common.Status.SUCCESS])
        return self.selectors_reward + self.constraints_reward + self.postcond_reward

    def init_evolution_algo(self):
        """Agent's GE algorithm operation defination."""
        # Genetic algorithm parameters
        self.operation_threshold = 50
        self.genome_storage = []

        # Grammatical Evolution part
        from ponyge.algorithm.parameters import Parameters
        parameter = Parameters()
        parameter_list = ['--parameters', '../..,nest.txt']
        # Comment when different results is desired.
        # Else set this for testing purpose
        # parameter.params['RANDOM_SEED'] = name
        # # np.random.randint(1, 99999999)
        # Set GE runtime parameters
        parameter.params['POPULATION_SIZE'] = self.operation_threshold // 2
        parameter.set_params(parameter_list)
        self.parameter = parameter
        # Initialize the genome
        individual = initialisation(self.parameter, size=3)
        individual = evaluate_fitness(individual, self.parameter)
        # Assign the genome to the agent
        self.individual = individual
        # Fitness
        self.beta = 0.9
        self.diversity_fitness = self.individual[0].fitness
        self.individual[0].fitness = 0
        self.generation = 0

        self.delayed_cf = 0
        self.delayed_ef = 0
        # if self.name == 7:
        #    self.individual[0].fitness = 10000000000000
        #    self.delayed_reward = 10000000000000

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
        # Reset the variabes
        self.debris_collected = 0
        self.location_history = set()
        self.no_cue_dropped = 0
        self.signal_time = 0
        self.signals = []
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

        # self.delayed_reward = round(self.beta * self.delayed_reward, 4)

        # # Goal Specification Fitness
        # self.individual[0].fitness = (1 - self.beta) * self.diversity_fitness + self.ef  + self.evaluate_constraints_conditions()
        divb = self.model.fmodels[self.model.fitid][0]
        efb = self.model.fmodels[self.model.fitid][1]
        # cfb = self.model.fmodels[self.model.fitid][2]
        dfb = self.model.fmodels[self.model.fitid][3]
        # print(self.name, divb, efb, dfb)

        self.individual[0].fitness = (
            (1 - self.beta) * self.delayed_reward * divb +
            self.ef * efb +
            # self.cf * cfb +
            # self.debris_collected * dfb
            self.evaluate_constraints_conditions() * dfb
            )
        # self.individual[0].fitness = (1 - self.beta) * self.delayed_reward + self.ef + self.evaluate_constraints_conditions()

    def get_debris_transported(self, distance_threshold=35):
        """Return debris that have been cleared from hub."""
        # Not computational efficient method
        # debris_objects = []
        # for debry in self.model.debris:
        #     distance = point_distance(debry.location, self.model.hub.location)
        #     if distance > distance_threshold:
        #         debris_objects.append(debry)
        debris_objects = []
        # debris_grid = []

        # grid = self.model.grid
        # for boundary in self.model.boundaries:
        #     # boundary_loc = boundary.location
        #     # neighbours = grid.get_neighborhood(boundary_loc, boundary.radius)
        #     # debris_objects += grid.get_objects_from_list_of_grid('Debris', neighbours)
        #     debris_objects +=  list(set(boundary.dropped_objects))
        #     # _, dgrid = grid.find_grid(boundary_loc)
        #     # debris_grid += dgrid

        # for debry in self.model.debris:
        #     _, debry_grid = grid.find_grid(debry.location)
        #     if debry_grid in debris_grid:
        #         debris_objects += [debry]
        # debris_objects = set(debris_objects)
        # return debris_objects
        debris_objects = list(set(self.model.boundary.dropped_objects))
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
        self.prev_location = copy.copy(self.location)
        self.geneticrate = False
        self.ltrate = 0
        # Increase beta
        # self.beta = self.timestamp / self.model.iter
        # if self.name ==1:
        # print(self.timestamp, self.name, len(self.trace))
        if self.dead is False:
            # Compute the behavior tree
            self.bt.behaviour_tree.tick()

            # Maintain location history
            _, gridval = self.model.grid.find_grid(self.location)
            self.location_history.add(gridval)

            # Hash the phenotype with its fitness
            # We need to move this from here to genetic step
            # self.cf = self.carrying_fitness()
            self.ef = self.exploration_fitness()
            # self.scf = self.communication_fitness()

            # Computes overall fitness using Beta function
            self.overall_fitness()

            self.phenotypes = dict()
            self.phenotypes[self.individual[0].phenotype] = (
                self.individual[0].fitness)

            if not self.model.stop_lateral_transfer:
                # Find the nearby agents
                cellmates = self.model.grid.get_objects_from_grid(
                    type(self).__name__, self.location)

                # Interaction Probability with other agents
                cellmates = [cell for cell in cellmates if self.model.random.rand() < self.model.iprob and cell.dead is False]
                # cellmates = [cell for cell in cellmates if cell.individual[0] not in self.genome_storage]
                self.ltrate = len(cellmates)
                # If neighbours found, store the genome
                if len(cellmates) > 1:
                    self.store_genome(cellmates)

            # Logic for gentic operations.
            # If the genome storage has enough genomes and agents has done some
            # exploration then compute the genetic step OR
            # 600 time step has passed and the agent has not done anything useful
            # then also perform genetic step
            storage_threshold = len(
                self.genome_storage) >= self.threshold

            if storage_threshold:
                self.geneticrate = storage_threshold
                self.genetic_step()
            elif (
                    (
                        storage_threshold is False and self.timestamp > self.model.gstep
                        ) and (self.exploration_fitness() < self.model.expp)):
                individual = initialisation(self.parameter, 10)
                individual = evaluate_fitness(individual, self.parameter)
                self.genome_storage = self.genome_storage + individual
                self.genetic_step()


class ExecutingAgent(NestAgent):
    """A Nest maintenance swarm agent.

    This agent will run the behaviors evolved.
    """

    def __init__(self, name, model, xmlstring=None):
        """Initialize the agent."""
        super().__init__(name, model)
        self.xmlstring = xmlstring
        self.blackboard = blackboard.Client(name=str(self.name))
        self.blackboard.register_key(key='neighbourobj', access=common.Access.WRITE)
        self.blackboard.neighbourobj = dict()

    def construct_bt(self):
        """Construct BT."""
        # Get the phenotype of the genome and store as xmlstring
        bts = []
        for i in range(len(self.xmlstring)):
            bt = BTConstruct(None, self, self.xmlstring[i])
            bt.construct()
            bts.append(bt.behaviour_tree.root)
        # root = Selector('RootAll')
        # root = Sequence('RootAll')
        root = Parallel('RootAll')
        self.model.random.shuffle(bts)
        root.add_children(bts)
        self.bt.behaviour_tree = BehaviourTree(root)

        # self.bt.xmlstring = self.xmlstring
        # # Construct actual BT from xmlstring
        # self.bt.construct()
        # py_trees.display.render_dot_tree(
        #    self.bt.behaviour_tree.root, name='/tmp/' + str(self.name))
        # py_trees.logging.level = py_trees.logging.Level.DEBUG
        # print(self.name, py_trees.display.ascii_tree(self.bt.behaviour_tree.root))

    def step(self):
        """Agent action at a single time step."""
        # Maintain the location history of the agent
        # self.location_history.add(self.location)

        # Compute the behavior tree
        self.bt.behaviour_tree.tick()

        # Find the no.of food collected from the BT execution
        # self.food_collected = len(self.get_food_in_hub())
