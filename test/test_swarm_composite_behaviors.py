"""Test files containging all the test cases for composit behaviors."""

from unittest import TestCase
from swarms.lib.agent import Agent
from swarms.lib.model import Model
from swarms.lib.time import SimultaneousActivation
from swarms.lib.space import Grid
from swarms.scbehaviors import MoveTowards, MoveAway
from swarms.sbehaviors import IsCarrying
from swarms.objects import Sites
import py_trees
import numpy as np


# Class to composite behaviors. For now there are 7 composite behaviors
class SwarmMoveTowards(Agent):
    """An minimalistic behavior tree for swarm agent implementing MoveTowards
    behavior using accleration and velocity
    """
    def __init__(self, name, model):
        super().__init__(name, model)
        self.location = ()

        self.direction = model.random.rand() * (2 * np.pi)
        # self.speed = 2
        self.radius = 3

        self.moveable = True
        self.shared_content = dict()

        self.shared_content['Sites'] = [model.target]

        # Defining the composite behavior
        movetowards = MoveTowards('MoveTowards')

        # Setup for the behavior
        movetowards.setup(0, self, 'Sites')

        # This behavior is just defined to check if the composite tree
        # can be combined with other primite behaviors
        is_carrying = IsCarrying('IsCarrying')
        is_carrying.setup(0, self, 'Food')

        seq = py_trees.composites.Sequence('Seq')
        seq.add_children([movetowards, is_carrying])

        # Since its root is a sequence, we can use it directly
        self.behaviour_tree = py_trees.trees.BehaviourTree(seq)

        # Debugging stuffs for py_trees
        py_trees.logging.level = py_trees.logging.Level.DEBUG
        py_trees.display.print_ascii_tree(movetowards)

    def step(self):
        self.behaviour_tree.tick()


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

        for i in range(self.num_agents):
            a = SwarmMoveTowards(i, self)
            self.schedule.add(a)
            x = -45
            y = -45
            a.location = (x, y)
            a.direction = -2.3561944901923448
            self.grid.add_object_to_grid((x, y), a)

        self.agent = a

    def step(self):
        self.schedule.step()


class TestGoToSwarmSmallGrid(TestCase):

    def setUp(self):
        self.environment = MoveTowardsModel(1, 100, 100, 10, 123)

        for i in range(68):
            self.environment.step()

    def test_agent_path(self):
        # Checking if the agents reaches site or not
        self.assertEqual(self.environment.agent.location, (45, 45))


class SwarmMoveAway(Agent):
    """An minimalistic behavior tree for swarm agent implementing MoveAway
    behavior.
    """
    def __init__(self, name, model):
        super().__init__(name, model)
        self.location = ()

        self.direction = model.random.rand() * (2 * np.pi)
        # self.speed = 2
        self.radius = 3

        self.moveable = True
        self.shared_content = dict()

        self.shared_content['Sites'] = [model.target]

        # Defining the composite behavior
        moveaway = MoveAway('MoveAway')

        # Setup for the behavior
        moveaway.setup(0, self, 'Sites')

        # This behavior is just defined to check if the composite tree
        # can be combined with other primite behaviors
        is_carrying = IsCarrying('IsCarrying')
        is_carrying.setup(0, self, 'Food')

        seq = py_trees.composites.Sequence('Seq')
        seq.add_children([moveaway, is_carrying])

        # Since its root is a sequence, we can use it directly
        self.behaviour_tree = py_trees.trees.BehaviourTree(seq)

        # Debugging stuffs for py_trees
        # py_trees.logging.level = py_trees.logging.Level.DEBUG
        # py_trees.display.print_ascii_tree(moveaway)

    def step(self):
        self.behaviour_tree.tick()


class MoveAwayModel(Model):
    """ A environment to model swarms """
    def __init__(self, N, width, height, grid=10, seed=None):
        if seed is None:
            super(MoveAwayModel, self).__init__(seed=None)
        else:
            super(MoveAwayModel, self).__init__(seed)

        self.num_agents = N

        self.grid = Grid(width, height, grid)

        self.schedule = SimultaneousActivation(self)

        self.target = Sites(id=1, location=(45, 45), radius=5, q_value=0.5)
        self.grid.add_object_to_grid(self.target.location, self.target)

        for i in range(self.num_agents):
            a = SwarmMoveAway(i, self)
            self.schedule.add(a)
            x = 15
            y = 15
            a.location = (x, y)
            a.direction = -2.3561944901923448
            self.grid.add_object_to_grid((x, y), a)

        self.agent = a

    def step(self):
        self.schedule.step()


class TestGoToAwaySwarmSmallGrid(TestCase):

    def setUp(self):
        self.environment = MoveAwayModel(1, 100, 100, 10, 123)

        for i in range(50):
            self.environment.step()

    def test_agent_path(self):
        # Checking if the agents reaches site or not
        self.assertEqual(self.environment.agent.location, (-42, -42))


