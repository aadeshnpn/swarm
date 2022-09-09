from inspect import CO_ITERABLE_COROUTINE
from py_trees.common import Status
from py_trees.composites import Selector, Sequence, Parallel
from py_trees import common, blackboard
from py_trees.trees import BehaviourTree
from swarms.behaviors.sbehaviors import (
    DropCue, SendSignal, ObjectsStore,
    NeighbourObjects)
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
# from flloat.parser.ltlf import LTLfParser
from collections import OrderedDict


class CoevoAgent(Agent):
    """An minimalistic Coevolution swarm agent."""

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
        self.ltrate = 0     # LT flag
        self.geneticrate = False
        self.brepotire = OrderedDict()

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

    def get_debris_transported(self):
        """Return debris that have been cleared from hub."""
        # Not computational efficient method
        debris_objects = []
        grid = self.model.grid
        for boundary in self.model.boundaries:
            boundary_loc = boundary.location
            neighbours = grid.get_neighborhood(boundary_loc, boundary.radius)
            debris_objects += grid.get_objects_from_list_of_grid('Debris', neighbours)
        debris_objects = set(debris_objects)

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
        # parser = LTLfParser()
        # parsed_formula = parser(goalspec)
        # result = parsed_formula.truth(trace)
        # return result
        pass


class LearningAgent(CoevoAgent):
    """Simple agent with GE capabilities."""

    def __init__(self, name, model, threshold=10):
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
        # self.trace.append({k:self.functions[k]() for k in self.keys})
        # self.trace[self.step_count] = {k:self.functions[k]() for k in self.keys}

    def update_brepotire(self):
        allnodes = list(self.bt.behaviour_tree.root.iterate())
        actions = list(filter(
            lambda x: x.name.split('_')[-1] == 'Act', allnodes)
            )
        if len(actions) == 1 and actions[0].status == Status.SUCCESS:
            if type(actions[0]).__name__ in ['MoveTowardsNormal', 'MoveAwayNormal']:
                action_key = type(actions[0]).__name__ + '_' + actions[0].item
            else:
                action_key = type(actions[0]).__name__
            val = self.brepotire.get(action_key, None)
            if val == None:
                self.brepotire[action_key] = self.individual[0]
            # else:
            #     if val.fitness < self.individual[0].fitness:
            #         self.brepotire[type(actions[0]).__name__] = self.individual[0]

    def update_brepotire_others(self, cell):
        allnodes = list(cell.bt.behaviour_tree.root.iterate())
        actions = list(filter(
            lambda x: x.name.split('_')[-1] == 'Act', allnodes)
            )
        # if actions[0].status == Status.SUCCESS:
        #     print(actions, actions[0].status)
        if len(actions) == 1 and actions[0].status == Status.SUCCESS:
            val = self.brepotire.get(type(actions[0]).__name__, None)
            if val is None:
                self.brepotire[type(actions[0]).__name__] = cell.individual[0]
            else:
                if val.fitness < cell.individual[0].fitness:
                    self.brepotire[type(actions[0]).__name__] = cell.individual[0]

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
        parameter_list = ['--parameters', '../..,ants.txt']
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

    def construct_bt(self):
        """Construct BT."""
        # Get the phenotype of the genome and store as xmlstring
        self.bt.xmlstring = self.individual[0].phenotype
        # Construct actual BT from xmlstring
        self.bt.construct()
        # print(self.bt.xmlstring)

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
        """Additional procedures called after genetic step."""

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

        # # Goal Specification Fitness
        self.individual[0].fitness = (1-self.beta) * self.delayed_reward + self.ef + self.evaluate_constraints_conditions()

    def get_food_in_hub(self, agent_name=True):
        """Get the food in the hub stored by the agent."""
        agent_food_objects = []
        food_objects = list(set(self.hub.dropped_objects))
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
            self.ef = self.exploration_fitness()

            # Computes overall fitness using Beta function
            self.overall_fitness()
            # print(self.name, self.individual[0].fitness)

            self.phenotypes = dict()
            self.phenotypes[self.individual[0].phenotype] = (
                self.individual[0].fitness)
            # Updated behavior repotire
            self.update_brepotire()
            if not self.model.stop_lateral_transfer:
                # Find the nearby agents
                cellmates = self.model.grid.get_objects_from_grid(
                    type(self).__name__, self.location)

                # Interaction Probability with other agents
                cellmates = [cell for cell in cellmates  if self.model.random.rand() < self.model.iprob and cell.dead is False]
                # cellmates = [cell for cell in cellmates if cell.individual[0] not in self.genome_storage]
                self.ltrate = len(cellmates)
                # If neighbours found, store the genome
                if len(cellmates) > 1:
                    # cellmates = list(self.model.random.choice(
                    #     cellmates, self.model.random.randint(
                    #         1, len(cellmates)-1), replace=False))
                    self.store_genome(cellmates)
                    # Update behavior repotire
                    # for cell in cellmates:
                    #     self.update_brepotire_others(cell)

            # Logic for gentic operations.
            # If the genome storage has enough genomes and agents has done some
            # exploration then compute the genetic step OR
            # 200 time step has passed and the agent has not done anything useful
            # then also perform genetic step
            storage_threshold = len(
                self.genome_storage) >= self.threshold
            # (self.model.num_agents / (self.threshold* 1.0))

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


class ExecutingAgent(CoevoAgent):
    """A coevolution swarm agent.

    This agent will run the various behaviors evolved.
    """

    def __init__(self, name, model, xmlstring=None):
        """Initialize the agent."""
        super().__init__(name, model)
        self.xmlstring = xmlstring
        self.blackboard = blackboard.Client(name=str(self.name))
        self.blackboard.register_key(key='neighbourobj', access=common.Access.WRITE)
        self.blackboard.neighbourobj = dict()
        self.current_behavior_counter = 0
        self.timer = 0


    def create_root_node(self, nodes):
        # root = Sequence('RootAll')
        # root = Selector('RootAll')
        # root = Parallel('RootAll')
        # self.model.random.shuffle(bts)
        # root.add_children(bts[self.current_behavior_counter])
        # root.add_children([node])
        # root.add_children(nodes)
        return BehaviourTree(nodes)

    def construct_bt(self):
        """Construct BT."""
        # Get the phenotype of the genome and store as xmlstring
        # self.bt.xmlstring = self.xmlstring
        # # Construct actual BT from xmlstring
        # self.bt.construct()

        # New way to create BT
        bts = []
        for i in range(len(self.xmlstring)):
            bt = BTConstruct(None, self, self.xmlstring[i])
            bt.construct()
            # bts.append(bt.behaviour_tree.root)
            bts.append(bt)
        self.bts = bts
        self.post_conditions = []
        for i in range(len(self.xmlstring)):
            bt = BTConstruct(None, self, self.xmlstring[i])
            bt.construct()
            # bts.append(bt.behaviour_tree.root)
            # bts.append(bt)
            other_branch_id = bt.behaviour_tree.root.children[0].children[1].id
            bt.behaviour_tree.prune_subtree(other_branch_id)
            self.post_conditions.append(bt)
        # root = Selector('RootAll')
        # root = Sequence('RootAll')
        # root = Parallel('RootAll')
        # self.model.random.shuffle(bts)
        # root.add_children(bts[self.current_behavior_counter])
        # self.bt.behaviour_tree = self.create_root_node(
        #     self.bts[self.current_behavior_counter])
        # self.current_behavior_counter += 1
        # print(self.bts, len(self.bts))

        # self.bt.behaviour_tree = self.create_root_node(
        #     self.bts)

        # Condition to change between different behaviors
        # If found any neighbours, then change behaviors
        # py_trees.display.render_dot_tree(
        #    self.bt.behaviour_tree.root, name='/tmp/' + str(self.name))
        # py_trees.logging.level = py_trees.logging.Level.DEBUG
        # for i in range(len(self.xmlstring)):
        # print(self.name, py_trees.display.ascii_tree(self.bts[0].behaviour_tree.root))
        # print(self.name, py_trees.display.ascii_tree(self.post_conditions[0].behaviour_tree.root))

        # print(self.bts[0].behaviour_tree.root.children[0].children[0].children)
        # postcondition_sequence = copy.copy(self.bts[0].behaviour_tree.root.children[0].children[0])
        # all_postcondition = []
        # for pconds in postcondition_sequence.children:
        #     pconds_copy = copy.copy(pconds)
        #     all_postcondition.append(pconds_copy)
        # postcondition_sequence_new = Sequence('PC_Sequence')
        # print(dir(pconds_copy))
        # postcondition_sequence_new.add_children(all_postcondition)
        # print('copied', postcondition_sequence_new.children)
        # exit()

    def check_conditions(self):
        cellobjects = self.model.grid.get_objects_from_grid(
                    None, self.location)
        object_names = ["Hub", "Sites", "Food"]
        objects_of_interest = [ cell for cell in cellobjects if type(cell).__name__ in object_names]
        return True if len(objects_of_interest) > 0 else False

    def check_post_condition(self):
        curr_postcondition = self.post_conditions[self.current_behavior_counter % len(self.xmlstring)]
        curr_postcondition.behaviour_tree.tick()
        # if self.name == 0:
        #     print(self.name, curr_postcondition.behaviour_tree.root.status, self.current_behavior_counter % len(self.xmlstring))
        if curr_postcondition.behaviour_tree.root.status == Status.SUCCESS:
            return True
        else:
            return False
        # post_cond = curr_bt


    def step(self):
        """Agent action at a single time step."""
        # Maintain the location history of the agent
        # self.location_history.add(self.location)

        # Compute the behavior tree
        self.bts[self.current_behavior_counter % len(self.xmlstring)].behaviour_tree.tick()

        # Check postcondition for that particular bt
        if self.check_post_condition():
            self.timer = 0
            self.current_behavior_counter += 1
        else:
            if self.timer > 10:
                self.timer = 0
                self.current_behavior_counter += 1
        self.timer += 1
        # If some condition meet change the behavior
        # if self.check_conditions():
        #     del self.bt.behaviour_tree
        #     self.bt.behaviour_tree = self.create_root_node(
        #         self.bts[self.current_behavior_counter % len(self.bts)])
        #     self.current_behavior_counter += 1

        # Find the no.of food collected from the BT execution
        # self.food_collected = len(self.get_food_in_hub())
