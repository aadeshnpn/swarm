from unittest import TestCase
from swarms.lib.agent import Agent
from swarms.lib.model import Model
from swarms.lib.time import SimultaneousActivation
from swarms.lib.space import Grid
from swarms.behaviors.scbehaviors import CompositeDrop
from swarms.lib.objects import Boundary, Debris, Hub, Food
import py_trees
# from py_trees.composites import RepeatUntilFalse
import numpy as np


# Class to test accleration and velocity models
class SwarmAgentDrop(Agent):
    """An minimalistic behavior tree for swarm agent implementing goto
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

        self.shared_content['Hub'] = {model.hub}
        # Vairables related to motion
        self.accleration = [0, 0]
        self.velocity = [0, 0]

        root = py_trees.composites.Sequence("Sequence")
        low = CompositeDrop('CDrop')
        low.setup(0, self, 'Food')
        root.add_children([low])
        self.behaviour_tree = py_trees.trees.BehaviourTree(root)
        py_trees.logging.level = py_trees.logging.Level.DEBUG
        # py_trees.display.print_ascii_tree(root)

    def step(self):
        self.behaviour_tree.tick()


class DropSwarmEnvironmentModel(Model):
    """ A environment to model swarms """
    def __init__(self, N, width, height, grid=10, seed=None):
        if seed is None:
            super(DropSwarmEnvironmentModel, self).__init__(seed=None)
        else:
            super(DropSwarmEnvironmentModel, self).__init__(seed)

        self.num_agents = N

        self.grid = Grid(width, height, grid)

        self.schedule = SimultaneousActivation(self)

        self.hub = Hub(radius=10)
        self.grid.add_object_to_grid(self.hub.location, self.hub)

        self.food = Food()
        self.grid.add_object_to_grid(self.food.location, self.food)

        self.debris = Debris()
        self.grid.add_object_to_grid(self.debris.location, self.debris)

        self.boundary = Boundary(location=(30, 30), radius=10)
        self.grid.add_object_to_grid(self.boundary.location, self.boundary)

        for i in range(1):
            a = SwarmAgentDrop(i, self)
            self.schedule.add(a)
            x = 30
            y = 30
            a.location = (x, y)
            a.direction = -2.3561944901923448
            self.grid.add_object_to_grid((x, y), a)
        self.agent = a
        self.agent.attached_objects.append(self.food)
        self.agent.model.grid.remove_object_from_grid(
            self.food.location, self.food)
        self.food.agent_name = self.agent.name

    def step(self):
        self.schedule.step()


class TestDropSwarmSmallGrid(TestCase):

    def setUp(self):
        self.environment = DropSwarmEnvironmentModel(1, 100, 100, 10, 123)

        for i in range(2):
            print(self.environment.agent.location)
            self.environment.step()

    def test_agent_path(self):
        self.assertEqual(self.environment.agent.location, (40, 40))
