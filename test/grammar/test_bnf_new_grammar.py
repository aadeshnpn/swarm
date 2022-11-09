"""Test files containging all the test cases for bnf grammar expansion."""

from unittest import TestCase
from swarms.lib.agent import Agent
from swarms.lib.model import Model
from swarms.lib.time import SimultaneousActivation
from swarms.lib.space import Grid
from swarms.behaviors.scbehaviors import (
    CompositeDrop, MoveTowards, MoveAway, Explore, CompositeSingleCarry
    )
from swarms.behaviors.sbehaviors import (
    IsCarrying, NeighbourObjects, Move, IsCarryable,
    SingleCarry, IsSingleCarry
    )
from swarms.lib.objects import Obstacles, Sites, Debris, Food, Traps, Hub
from py_trees.trees import BehaviourTree
from py_trees.behaviour import Behaviour
from py_trees.composites import Sequence, Selector, Parallel
from py_trees import common, blackboard
from py_trees.decorators import FailureIsSuccess
import py_trees
import numpy as np

from ponyge.operators.initialisation import initialisation
from ponyge.fitness.evaluation import evaluate_fitness
from ponyge.operators.crossover import crossover
from ponyge.operators.mutation import mutation
from ponyge.operators.replacement import replacement
from ponyge.operators.selection import selection
from swarms.utils.bt import BTComplexConstruct, BTConstruct


class SwarmMoveTowards(Agent):
    """An minimalistic behavior tree for swarm agent implementing MoveTowards
    behavior using accleration and velocity
    """
    def __init__(self, name, model):
        super().__init__(name, model)
        self.location = ()

        self.direction = 0.5
        # self.speed = 2
        self.radius = 3

        self.moveable = True
        self.shared_content = dict()

        # Grammatical Evolution part
        from ponyge.algorithm.parameters import Parameters
        parameter = Parameters()
        parameter_list = ['--parameters', '../..,test_new.txt']
        parameter.params['RANDOM_SEED'] = 1235
        parameter.params['POPULATION_SIZE'] = 10 // 2
        parameter.set_params(parameter_list)
        self.parameter = parameter
        individual = initialisation(self.parameter, 1)
        individual = evaluate_fitness(individual, self.parameter)
        # self.mapper.xmlstring = self.individual.phenotype
        print(self.name, 'test new',  individual[0].phenotype)
        self.individual = individual
        # print(individual[0])
        self.bt = BTConstruct(None, self)
        self.bt.xmlstring = self.individual[0].phenotype
        # Construct actual BT from xmlstring
        self.bt.construct()
        # # Debugging stuffs for py_trees
        py_trees.logging.level = py_trees.logging.Level.DEBUG
        print(py_trees.display.ascii_tree(self.bt.behaviour_tree.root))
        # allnodes = list(self.bt.behaviour_tree.root.iterate())
        # print([a.name for a in allnodes])
        # self.blackboard = blackboard.Client(name=str(name))
        # self.blackboard.register_key(key='neighbourobj', access=common.Access.WRITE)
        # self.blackboard.neighbourobj = dict()

    def btfitness(self):
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
        return (selectors_reward, constraints_reward, postcond_reward)

    def step(self):
        pass
        # self.bt.behaviour_tree.tick()


class MoveTowardsModel(Model):
    """ A environment to model swarms """
    def __init__(self, N, width, height, grid=10, seed=None):
        if seed is None:
            super(MoveTowardsModel, self).__init__(seed=None)
        else:
            super(MoveTowardsModel, self).__init__(seed)

        self.num_agents = N

        self.grid = Grid(width, height, grid)

        self.schedule = SimultaneousActivation(self)

        self.target = Sites(id=1, location=(45, 45), radius=5, q_value=0.5)
        self.grid.add_object_to_grid(self.target.location, self.target)
        self.agents = []
        for i in range(self.num_agents):
            a = SwarmMoveTowards(i, self)
            self.schedule.add(a)
            x = -45
            y = -45
            a.location = (x, y)
            a.direction = -2.3561944901923448
            self.grid.add_object_to_grid((x, y), a)
            self.agents.append(a)
        self.agent = a

    def step(self):
        self.schedule.step()


class TestGoToSwarmSmallGrid(TestCase):

    def setUp(self):
        self.environment = MoveTowardsModel(1, 100, 100, 10, 123)

        for i in range(1):
            self.environment.step()

    def test_agent_path(self):
        # Checking if the agents reaches site or not
        self.assertEqual(len(list(self.environment.agent.bt.behaviour_tree.root.iterate())), 29)


# class TestVisitorSwarmSmallGrid(TestCase):

#     def setUp(self):
#         self.environment = MoveTowardsModel(1, 100, 100, 10, 123)

#         for i in range(1):
#             self.environment.step()

#     def test_agent_fitness(self):
#         self.assertEqual(self.environment.agent.btfitness(), (4, 0, 0))


# class TestDiversityFitnessSwarmSmallGrid(TestCase):

#     def setUp(self):
#         self.environment = MoveTowardsModel(1, 100, 100, 10, 123)

#         for i in range(1):
#             self.environment.step()

#     def test_agent_diversity(self):
#         # Checking if the agents reaches site or not
#         # self.assertEqual(self.environment.agent.btfitness(), (1, 0, 0))
#         self.assertEqual(self.environment.agent.individual[0].fitness, 12.5)


# def main():
#     environment = MoveTowardsModel(1, 100, 100, 10, None)

#     for i in range(1):
#         environment.step()

#     print(environment.agent.btfitness())
#     # print(environment.agent.individual[0].fitness)
#     for a in environment.agents:
#         print(a.individual[0].fitness)
#         print(a.bt.behaviour_tree.root)
#         print(a.individual[0].phenotype)
#         print(py_trees.display.ascii_tree(a.bt.behaviour_tree.root))

# main()