from inspect import CO_ITERABLE_COROUTINE
from py_trees import common
from py_trees.composites import Selector
from py_trees import common, blackboard
from swarms.behaviors.sbehaviors import DropCue, SendSignal, ObjectsStore
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
import copy
from flloat.parser.ltlf import LTLfParser


class ForagingAgent(Agent):
    """An minimalistic foraging swarm agent."""

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

        # Goal related attributes
        # P is prospective, C is communication
        # T is trap, and O is obstacle
        # self.keys = ['p', 'c', 't', 'o']
        # self.keys = ['o', 'e', 'p']
        # iscarrying, issignalactive, avoided trap, avoided obstacle, food dropped at hub
        self.keys = ['c', 's', 'at', 'ao', 'f']
        self.goalspecs = {
            self.keys[0]: 'c',
            self.keys[1]: 's',
            self.keys[2]: 'at',
            self.keys[3]: 'ao',
            self.keys[4]: 'f'
            }
        # self.trace = [{k:list() for k in self.keys}]
        # self.trace = [None] * (model.iter+1)

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

    def get_food_in_hub(self, agent_name=True):
        """Get the food in the hub stored by the agent."""
        grid = self.model.grid
        hub_loc = self.model.hub.location
        neighbours = grid.get_neighborhood(hub_loc, self.model.hub.radius)
        food_objects = grid.get_objects_from_list_of_grid('Food', neighbours)
        agent_food_objects = []
        if not agent_name:
            for food in food_objects:
                agent_food_objects.append(food)
        else:
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


class LearningAgent(ForagingAgent):
    """Simple agent with GE capabilities."""

    def __init__(self, name, model):
        """Initialize the agent."""
        super().__init__(name, model)
        self.delayed_reward = 0
        self.phenotypes = dict()
        self.functions = {
            self.keys[0]: self.proposition_c,
            self.keys[1]: self.proposition_s,
            self.keys[2]: self.proposition_at,
            self.keys[3]: self.proposition_ao,
            self.keys[4]: self.proposition_f
            }
        # Flags to make the computation easier
        self.avoid_trap = False
        self.avoid_obs = False
        self.blackboard = blackboard.Client(name=str(self.name))
        self.blackboard.register_key(key='neighbourobj', access=common.Access.WRITE)

        # self.trace.append({k:self.functions[k]() for k in self.keys})
        # self.trace[self.step_count] = {k:self.functions[k]() for k in self.keys}

    def proposition_c(self):
        if len(self.attached_objects) > 0:
            return True
        else:
            return False

    def proposition_s(self):
        if len(self.signals) > 0:
            return True
        else:
            return False

    def proposition_at(self):
        return self.avoid_trap

    def proposition_ao(self):
        return self.avoid_obs

    def proposition_f(self):
        grid = self.model.grid
        hub_loc = self.model.hub.location
        neighbours = grid.get_neighborhood(hub_loc, self.model.hub.radius)
        food_objects = grid.get_objects_from_list_of_grid('Food', neighbours)
        agent_food_objects = []
        prop_o = False
        for food in food_objects:
            # print('p keys', self.name, food.phenotype.keys())
            if (
                food.agent_name == self.name and (
                    self.individual[0].phenotype in list(
                        food.phenotype.keys())
                    )):
                prop_o = True
                break
        return prop_o

    def evaluate_goals(self):
        goals = []
        # print('from evaluate goals', self.trace)
        trace = list(filter(None.__ne__, self.trace))
        trace = [trace[-1]]
        for key, value in self.goalspecs.items():
            formula = self.model.parser(value)
            goals += [formula.truth(trace)]
            # if key == 'e':
            #     goals += [formula.truth(trace, len(trace)-1)]
            # else:
            #    goals += [formula.truth(trace)]
        #     if self.name ==1:
        #         print(key, value)
        # if self.name ==1:
        # print('from evalutate goals', self.name, goals, end=' ')
        goals += [1 if self.bt.behaviour_tree.root.status == py_trees.Status.SUCCESS else 0]
        return sum(goals)

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
        # print(list(self.bt.behaviour_tree.visitors))
        selectors_reward = sum([1 for sel in selectors if sel.status == common.Status.SUCCESS])
        constraints_reward = sum([-2 for const in constraints if const.status == common.Status.FAILURE])
        postcond_reward = sum([1 for pcond in postcond if pcond.status == common.Status.SUCCESS])
        return selectors_reward + constraints_reward + postcond_reward

    def init_evolution_algo(self):
        """Agent's GE algorithm operation defination."""
        # Genetic algorithm parameters
        self.operation_threshold = 50
        self.genome_storage = []

        # Grammatical Evolution part
        from ponyge.algorithm.parameters import Parameters
        parameter = Parameters()
        parameter_list = ['--parameters', '../..,res.txt']
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
        # for agent in cellmates:
        #    if agent.food_collected > 0:
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
        # Reset the variabes
        self.food_collected = 0
        self.location_history = set()
        self.no_cue_dropped = 0
        self.signal_time = 0
        self.signals = []
        # self.trace = [None] * self.model.iter
        # self.trace.append({k:self.functions[k]() for k in self.keys})
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

        # self.individual[0].fitness = (
        #     self.ef + self.cf * 4 + self.food_collected * 8)

        # self.individual[0].fitness = (1 - self.beta) * self.delayed_reward \
        #     + self.ef + self.cf \
        #     + self.food_collected

        # Goal Specification Fitness
        self.individual[0].fitness = (1 - self.beta) * self.delayed_reward \
            + self.ef + self.evaluate_constraints_conditions()


    def get_food_in_hub(self, agent_name=True):
        """Get the food in the hub stored by the agent."""
        grid = self.model.grid
        hub_loc = self.model.hub.location
        neighbours = grid.get_neighborhood(hub_loc, self.model.hub.radius)
        food_objects = grid.get_objects_from_list_of_grid('Food', neighbours)
        agent_food_objects = []
        if not agent_name:
            for food in food_objects:
                agent_food_objects.append(food.weight)
        else:
            for food in food_objects:
                # print('p keys', self.name, food.phenotype.keys())
                if (
                    food.agent_name == self.name and (
                        self.individual[0].phenotype in list(
                            food.phenotype.keys())
                        )):
                    agent_food_objects.append(food.weight)
        return sum(agent_food_objects)

    def step(self):
        """Take a step in the simulation."""
        # py_trees.logging.level = py_trees.logging.Level.DEBUG
        # output = py_trees.display.ascii_tree(self.bt.behaviour_tree.root)

        # Couting variables
        self.timestamp += 1
        self.step_count += 1
        self.prev_location = copy.copy(self.location)
        # Increase beta
        # self.beta = self.timestamp / self.model.iter
        # if self.name ==1:
        # print(self.timestamp, self.name, len(self.trace))

        # Compute the behavior tree
        self.bt.behaviour_tree.tick()

        # Maintain location history
        _, gridval = self.model.grid.find_grid(self.location)
        self.location_history.add(gridval)

        # Add to trace
        # self.trace.append({k:self.functions[k]() for k in self.keys})
        # print(len(self.trace))
        # self.trace[self.step_count] = {k:self.functions[k]() for k in self.keys}

        # Find the no.of food collected from the BT execution
        self.food_collected = self.get_food_in_hub()  # * self.get_food_in_hub(
        #  False)

        # Hash the phenotype with its fitness
        # We need to move this from here to genetic step
        # self.cf = self.carrying_fitness()
        self.ef = self.exploration_fitness()
        # self.scf = self.communication_fitness()

        # Computes overall fitness using Beta function
        self.overall_fitness()
        # print(self.name, self.individual[0].fitness)
        # Debugging
        # decodedata = "b\'" + self.individual[0].phenotype + "\'"
        # encode = self.individual[0].phenotype.encode('utf-8')
        # print(
        #    self.name, hashlib.sha224(encode).hexdigest(
        #    ), self.food_collected, cf, ef)
        """
        if self.individual[0].phenotype in self.phenotypes.keys():
            e, c, f = self.phenotypes[self.individual[0].phenotype]
            if f < self.food_collected:
                f = self.food_collected
            if c < self.cf:
                c = self.cf
            if e < self.ef:
                e = self.ef

            self.phenotypes[self.individual[0].phenotype] = (e, c, f)
        else:
            # if int(cf) == 0 and int(ef) == 0
            # and int(self.food_collected) == 0:
            if int(self.cf) == 0 and int(self.food_collected) == 0:
                pass
            else:
                self.phenotypes[self.individual[0].phenotype] = (
                    self.ef, self.cf, self.food_collected)
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
                    ) and self.timestamp > 200 and self.food_collected <= 0):
            self.genetic_step()
        """
        if storage_threshold:
            self.genetic_step()
        elif (
                (
                    storage_threshold is False and self.timestamp > 100
                    ) and (self.exploration_fitness() < 2)):
            individual = initialisation(self.parameter, 10)
            individual = evaluate_fitness(individual, self.parameter)
            self.genome_storage = self.genome_storage + individual
            self.genetic_step()
        """


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
