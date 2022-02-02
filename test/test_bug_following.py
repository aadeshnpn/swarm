from unittest import TestCase
from swarms.lib.agent import Agent
from swarms.lib.model import Model
from swarms.lib.time import SimultaneousActivation
from swarms.lib.space import Grid
from swarms.behaviors.sbehaviors import (
    GoTo, RandomWalk, NeighbourObjects,
    Away, Towards, DoNotMove, Move, AvoidSObjects
    )

from swarms.behaviors.scbehaviors import (
    MoveTowards,
    MoveAway, Explore
    )
from swarms.lib.objects import Sites, Hub, Obstacles, Traps
import py_trees
from py_trees import common, blackboard
import numpy as np
from py_trees.composites import Sequence, Selector, Parallel
from py_trees.trees import BehaviourTree


# Class to tets Passable attribute for agents
class SwarmAgentGoTo(Agent):
    """ An minimalistic behavior tree for swarm agent implementing goto
    behavior
    """
    def __init__(self, name, model):
        super().__init__(name, model)
        self.location = ()

        self.direction = model.random.rand() * (2 * np.pi)
        self.speed = 2
        self.radius = 5

        self.moveable = True
        self.shared_content = dict()

        root = py_trees.composites.Sequence("Sequence")


        self.shared_content = dict()
        self.shared_content[type(model.target).__name__] = {model.target}

        # Defining the composite behavior
        movetowards = MoveTowards('MoveTowards')

        # Setup for the behavior
        movetowards.setup(0, self, 'Sites')

        # This behavior is just defined to check if the composite tree
        # can be combined with other primite behaviors
        seq = Sequence('Seq')
        seq.add_children([movetowards])

        # Since its root is a sequence, we can use it directly
        self.behaviour_tree = BehaviourTree(seq)

    def step(self):
        self.behaviour_tree.tick()


class GoToSwarmEnvironmentModel(Model):
    """ A environemnt to model swarms """
    def __init__(self, N, width, height, grid=10, seed=None):
        if seed is None:
            super(GoToSwarmEnvironmentModel, self).__init__(seed=None)
        else:
            super(GoToSwarmEnvironmentModel, self).__init__(seed)

        self.num_agents = N

        self.grid = Grid(width, height, grid)

        self.schedule = SimultaneousActivation(self)

        self.obstacle = Obstacles(id=2, location=(9, 9), radius=5)

        self.target = Sites(id=1, location=(45, 45), radius=5, q_value=0.5)

        self.grid.add_object_to_grid(self.target.location, self.target)
        self.grid.add_object_to_grid(self.obstacle.location, self.obstacle)

        for i in range(self.num_agents):
            a = SwarmAgentGoTo(i, self)
            self.schedule.add(a)
            x = -45
            y = -30
            a.location = (x, y)
            a.direction = -2.3561944901923448
            self.grid.add_object_to_grid((x, y), a)

        self.agent = a

    def step(self):
        self.schedule.step()


class TestGoToSwarmSmallGrid(TestCase):

    def setUp(self):
        self.environment = GoToSwarmEnvironmentModel(1, 100, 100, 10, 123)

        for i in range(72):
            self.environment.step()
            # print(i, self.environment.agent.location)

    def test_agent_path(self):
        self.assertEqual(self.environment.agent.location, (40, 40))

    def test_agent_grid(self):
        self.assertIsInstance(
            self.environment.grid.get_objects_from_grid('Sites',self.environment.agent.location)[0], Sites)
