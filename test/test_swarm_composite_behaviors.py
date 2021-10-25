"""Test files containging all the test cases for composit behaviors."""

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

        self.shared_content['Sites'] = {model.target}

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

        # Debugging stuffs for py_trees
        # py_trees.logging.level = py_trees.logging.Level.DEBUG
        # print(py_trees.display.ascii_tree(movetowards))

    def step(self):
        self.behaviour_tree.tick()


# Class to composite behaviors. For now there are 7 composite behaviors
class MoveTowardsAgent(Agent):
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

        self.shared_content['Sites'] = {model.target}

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

        # Debugging stuffs for py_trees
        # py_trees.logging.level = py_trees.logging.Level.DEBUG
        # print(py_trees.display.ascii_tree(movetowards))

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
        self.assertEqual(self.environment.agent.location, (23, 23))


class MoveTowardsModelObs(Model):
    """ A environment to model swarms """
    def __init__(self, N, width, height, grid=10, seed=None):
        if seed is None:
            super(MoveTowardsModelObs, self).__init__(seed=None)
        else:
            super(MoveTowardsModelObs, self).__init__(seed)

        self.num_agents = N

        self.grid = Grid(width, height, grid)

        self.schedule = SimultaneousActivation(self)

        self.target = Sites(id=1, location=(45, 45), radius=5, q_value=0.5)
        self.grid.add_object_to_grid(self.target.location, self.target)

        self.obstacle = Obstacles(id=2, location=(5, 0), radius=5)
        self.grid.add_object_to_grid(self.obstacle.location, self.obstacle)

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


class TestGoToObsSwarmSmallGrid(TestCase):

    def setUp(self):
        self.environment = MoveTowardsModelObs(1, 100, 100, 10, 123)

        for i in range(75):
            self.environment.step()

    def test_agent_path(self):
        # Checking if the agents reaches site or not
        self.assertEqual(self.environment.agent.location, (36, 35))


class MoveTowardsModelObsTrap(Model):
    """ A environment to model swarms """
    def __init__(self, N, width, height, grid=10, seed=None):
        if seed is None:
            super(MoveTowardsModelObsTrap, self).__init__(seed=None)
        else:
            super(MoveTowardsModelObsTrap, self).__init__(seed)

        self.num_agents = N

        self.grid = Grid(width, height, grid)

        self.schedule = SimultaneousActivation(self)

        self.target = Sites(id=1, location=(45, 45), radius=5, q_value=0.5)
        self.grid.add_object_to_grid(self.target.location, self.target)

        self.obstacle = Obstacles(id=2, location=(5, 0), radius=5)
        self.grid.add_object_to_grid(self.obstacle.location, self.obstacle)

        self.trap = Traps(id=2, location=(12, 22), radius=5)
        self.grid.add_object_to_grid(self.trap.location, self.trap)

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


class TestGoToObsTrapSwarmSmallGrid(TestCase):

    def setUp(self):
        self.environment = MoveTowardsModelObsTrap(1, 100, 100, 10, 123)

        for i in range(100):
            self.environment.step()

    def test_agent_path(self):
        # Checking if the agents reaches site or not
        self.assertEqual(self.environment.agent.location, (40, 40))


class MoveTowardsModelObsTrapBig(Model):
    """ A environment to model swarms """
    def __init__(self, N, width, height, grid=10, seed=None):
        if seed is None:
            super(MoveTowardsModelObsTrapBig, self).__init__(seed=None)
        else:
            super(MoveTowardsModelObsTrapBig, self).__init__(seed)

        self.num_agents = N

        self.grid = Grid(width, height, grid)

        self.schedule = SimultaneousActivation(self)

        self.target = Sites(id=1, location=(45, 45), radius=5, q_value=0.5)
        self.grid.add_object_to_grid(self.target.location, self.target)

        # self.obstacle = Obstacles(id=2, location=(5, 0), radius=20)
        # self.grid.add_object_to_grid(self.obstacle.location, self.obstacle)

        self.trap = Traps(id=2, location=(15, 15), radius=15)
        self.grid.add_object_to_grid(self.trap.location, self.trap)

        # self.agents = []
        for i in range(self.num_agents):
            a = MoveTowardsAgent(i, self)
            self.schedule.add(a)
            x = -5
            y = -5
            a.location = (x, y)
            a.direction = -2.3561944901923448
            self.grid.add_object_to_grid((x, y), a)
            self.agent = a

    def step(self):
        self.schedule.step()


class TestGoToObsTrapBigSwarmSmallGrid(TestCase):

    def setUp(self):
        self.environment = MoveTowardsModelObsTrapBig(1, 100, 100, 10, 123)

        for i in range(80):
            self.environment.step()
            # print(i, self.environment.agent.location, self.environment.agent.dead)

    def test_agent_path(self):
        # Checking if the agents reaches site or not
        self.assertEqual(self.environment.agent.location, (44, 40))


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

        self.shared_content['Sites'] = {model.target}

        # Defining the composite behavior
        moveaway = MoveAway('MoveAway')

        # Setup for the behavior
        moveaway.setup(0, self, 'Sites')

        seq = Sequence('Seq')
        seq.add_children([moveaway])

        # Since its root is a sequence, we can use it directly
        self.behaviour_tree = BehaviourTree(seq)

        # Debugging stuffs for py_trees
        # py_trees.logging.level = py_trees.logging.Level.DEBUG
        # py_trees.display.ascii_tree(moveaway)

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
        self.assertEqual(self.environment.agent.location, (-35, -35))


class MoveAwayObsModel(Model):
    """ A environment to model swarms """
    def __init__(self, N, width, height, grid=10, seed=None):
        if seed is None:
            super(MoveAwayObsModel, self).__init__(seed=None)
        else:
            super(MoveAwayObsModel, self).__init__(seed)

        self.num_agents = N

        self.grid = Grid(width, height, grid)

        self.schedule = SimultaneousActivation(self)

        self.target = Sites(id=1, location=(45, 45), radius=5, q_value=0.5)
        self.grid.add_object_to_grid(self.target.location, self.target)

        self.obstacle = Obstacles(id=2, location=(5, 0), radius=11)
        self.grid.add_object_to_grid(self.obstacle.location, self.obstacle)

        for i in range(self.num_agents):
            a = SwarmMoveAway(i, self)
            self.schedule.add(a)
            x = 25
            y = 25
            a.location = (x, y)
            a.direction = -2.3561944901923448
            self.grid.add_object_to_grid((x, y), a)

        self.agent = a

    def step(self):
        self.schedule.step()


class TestGoToAwayObsSwarmSmallGrid(TestCase):

    def setUp(self):
        self.environment = MoveAwayObsModel(1, 100, 100, 10, 123)

        for i in range(50):
            self.environment.step()
            # print(self.environment.agent.location)

    def test_agent_path(self):
        # Checking if the agents reaches site or not
        self.assertEqual(self.environment.agent.location, (-48, -8))


class MoveAwayObsTrapModel(Model):
    """ A environment to model swarms """
    def __init__(self, N, width, height, grid=10, seed=None):
        if seed is None:
            super(MoveAwayObsTrapModel, self).__init__(seed=None)
        else:
            super(MoveAwayObsTrapModel, self).__init__(seed)

        self.num_agents = N

        self.grid = Grid(width, height, grid)

        self.schedule = SimultaneousActivation(self)

        self.target = Sites(id=1, location=(45, 45), radius=5, q_value=0.5)
        self.grid.add_object_to_grid(self.target.location, self.target)

        self.obstacle = Obstacles(id=2, location=(5, 0), radius=11)
        self.grid.add_object_to_grid(self.obstacle.location, self.obstacle)

        self.trap = Traps(id=2, location=(-20, -15), radius=11)
        self.grid.add_object_to_grid(self.trap.location, self.trap)

        for i in range(self.num_agents):
            a = SwarmMoveAway(i, self)
            self.schedule.add(a)
            x = 25
            y = 25
            a.location = (x, y)
            a.direction = -2.3561944901923448
            self.grid.add_object_to_grid((x, y), a)

        self.agent = a

    def step(self):
        self.schedule.step()


class TestGoToAwayObsTrapSwarmSmallGrid(TestCase):

    def setUp(self):
        self.environment = MoveAwayObsTrapModel(1, 100, 100, 10, 123)

        for i in range(50):
            # print(self.environment.agent.location)
            self.environment.step()

    def test_agent_path(self):
        # Checking if the agents reaches site or not
        self.assertEqual(self.environment.agent.location, (-48, -8))


# # class to define agent explore behavior
class SwarmExplore(Agent):
    """ An minimalistic behavior tree for swarm agent
    implementing Explore
    """
    def __init__(self, name, model):
        super().__init__(name, model)
        self.wealth = 1
        self.location = ()

        self.direction = model.random.rand() * (2 * np.pi)
        self.speed = 2
        self.radius = 3

        self.moveable = True
        self.shared_content = dict()

        # Defining the composite behavior
        explore = Explore('Explore')

        # Setup for the behavior
        explore.setup(0, self, None)

        seq = Sequence('Seq')
        seq.add_children([explore])

        # Since its root is a sequence, we can use it directly
        self.behaviour_tree = BehaviourTree(seq)

    def step(self):
        self.behaviour_tree.tick()

    def advance(self):
        pass


class ExploreModel(Model):
    """ A environemnt to model swarms """
    def __init__(self, N, width, height, grid=10, seed=None):
        if seed is None:
            super(ExploreModel, self).__init__(seed=None)
        else:
            super(ExploreModel, self).__init__(seed)

        self.num_agents = N

        self.grid = Grid(width, height, grid)

        self.schedule = SimultaneousActivation(self)

        for i in range(self.num_agents):
            a = SwarmExplore(i, self)
            self.schedule.add(a)
            x = 0
            y = 0
            a.location = (x, y)
            a.direction = -2.3561944901923448
            self.grid.add_object_to_grid((x, y), a)

        self.agent = a

    def step(self):
        self.schedule.step()


class TestExploreSwarmSmallGrid(TestCase):

    def setUp(self):
        self.environment = ExploreModel(
            1, 100, 100, 10, 123)

        location_results = []

        for i in range(50):
            location_results.append(self.environment.agent.location)
            self.environment.step()

        self.trimmed_results = location_results[0:2] + location_results[47:]

    def test_agent_path(self):
        self.assertEqual(self.trimmed_results, [
            (0, 0), (-2, -1), (-32, -48), (-30, -48), (-28, -48)])


class ExploreObsModel(Model):
    """ A environemnt to model swarms """
    def __init__(self, N, width, height, grid=10, seed=None):
        if seed is None:
            super(ExploreObsModel, self).__init__(seed=None)
        else:
            super(ExploreObsModel, self).__init__(seed)

        self.num_agents = N

        self.grid = Grid(width, height, grid)

        self.obstacle = Obstacles(id=2, location=(-25, -25), radius=11)
        self.grid.add_object_to_grid(self.obstacle.location, self.obstacle)

        self.schedule = SimultaneousActivation(self)

        for i in range(self.num_agents):
            a = SwarmExplore(i, self)
            self.schedule.add(a)
            x = 0
            y = 0
            a.location = (x, y)
            a.direction = -2.3561944901923448
            self.grid.add_object_to_grid((x, y), a)

        self.agent = a

    def step(self):
        self.schedule.step()


class TestExploreObsSwarmSmallGrid(TestCase):

    def setUp(self):
        self.environment = ExploreObsModel(
            1, 100, 100, 10, 123)

        location_results = []

        for i in range(50):
            location_results.append(self.environment.agent.location)
            self.environment.step()

        self.trimmed_results = location_results[0:2] + location_results[47:]

    def test_agent_path(self):
        self.assertEqual(self.trimmed_results, [
            (0, 0), (-2, -1), (0, 46), (-2, 45), (-4, 44)])


class SwarmSingleCarry(Agent):
    """An minimalistic behavior tree for swarm agent implementing
    CompositeCarry behavior
    """
    def __init__(self, name, model):
        super().__init__(name, model)
        self.location = ()

        self.direction = model.random.rand() * (2 * np.pi)
        # self.speed = 2
        self.radius = 3

        self.moveable = True
        self.shared_content = dict()

        root = Sequence("Sequence")

        # Creating composite single carry object
        singlecarry = CompositeSingleCarry('SingleCarry')
        singlecarry.setup(0, self, 'Debris')

        root.add_children([singlecarry])
        self.behaviour_tree = BehaviourTree(root)

        # Debugging stuffs for py_trees
        # py_trees.logging.level = py_trees.logging.Level.DEBUG
        # py_trees.display.print_ascii_tree(root)

    def step(self):
        self.behaviour_tree.tick()


class SingleCarryModel(Model):
    """ A environment to model swarms """
    def __init__(self, N, width, height, grid=10, seed=None):
        if seed is None:
            super(SingleCarryModel, self).__init__(seed=None)
        else:
            super(SingleCarryModel, self).__init__(seed)

        self.num_agents = N

        self.grid = Grid(width, height, grid)

        self.schedule = SimultaneousActivation(self)

        self.thing = Debris(id=1, location=(0, 0), radius=4)
        self.grid.add_object_to_grid(self.thing.location, self.thing)

        for i in range(self.num_agents):
            a = SwarmSingleCarry(i, self)
            self.schedule.add(a)
            x = 0
            y = 0
            a.location = (x, y)
            a.direction = -2.3561944901923448
            self.grid.add_object_to_grid((x, y), a)

        self.agent = a

    def step(self):
        self.schedule.step()


class TestSingleCarry(TestCase):

    def setUp(self):
        self.environment = SingleCarryModel(
            1, 100, 100, 10, 123)

        for i in range(2):
            self.environment.step()

    def test_agent_path(self):
        self.assertEqual(
            self.environment.agent.attached_objects[0], self.environment.thing)


class SwarmSingleCarryDrop(Agent):
    """An minimalistic behavior tree for swarm agent implementing
    CompositeCarry behavior and CompositeDrop behavior
    """
    def __init__(self, name, model):
        super().__init__(name, model)
        self.location = ()

        self.direction = model.random.rand() * (2 * np.pi)
        # self.speed = 2
        self.radius = 3

        self.moveable = True
        self.shared_content = dict()

        name = type(model.hub).__name__
        self.shared_content[name] = {model.hub}

        root = Sequence("Sequence")

        # Creating composite single carry object
        singlecarry = CompositeSingleCarry('SingleCarry')
        singlecarry.setup(0, self, 'Debris')

        drop = CompositeDrop('Drop')
        drop.setup(0, self, 'Debris')

        movetowards = MoveTowards('MoveTowardsHub')
        movetowards.setup(0, self, 'Hub')

        root.add_children([singlecarry, movetowards, drop])
        self.behaviour_tree = BehaviourTree(root)

        # Debugging stuffs for py_trees
        # py_trees.logging.level = py_trees.logging.Level.DEBUG
        # print(py_trees.display.ascii_tree(root))

    def step(self):
        self.behaviour_tree.tick()


class SingleCarryDropModel(Model):
    """ A environment to model swarms """
    def __init__(self, N, width, height, grid=10, seed=None):
        if seed is None:
            super(SingleCarryDropModel, self).__init__(seed=None)
        else:
            super(SingleCarryDropModel, self).__init__(seed)

        self.num_agents = N

        self.grid = Grid(width, height, grid)

        self.schedule = SimultaneousActivation(self)

        self.thing = Debris(id=1, location=(30, 30), radius=4)
        self.grid.add_object_to_grid(self.thing.location, self.thing)

        # self.target = Sites(id=1, location=(5, 5), radius=8, q_value=0.5)
        # self.grid.add_object_to_grid(self.target.location, self.target)

        self.hub = Hub(id=1, location=(-20, -20), radius=9)
        self.grid.add_object_to_grid(self.hub.location, self.hub)

        for i in range(self.num_agents):
            a = SwarmSingleCarryDrop(i, self)
            self.schedule.add(a)
            x = 30
            y = 30
            a.location = (x, y)
            a.direction = -2.3561944901923448
            self.grid.add_object_to_grid((x, y), a)

        self.agent = a

    def step(self):
        self.schedule.step()


class TestSingleCarryDrop(TestCase):

    def setUp(self):
        self.environment = SingleCarryDropModel(
            1, 100, 100, 10, 123)

        for i in range(42):
            print(self.environment.agent.location)
            self.environment.step()

    def test_agent_path(self):
        self.assertEqual(
            self.environment.agent.location, (-11, -11))

    def test_agent_dropped(self):
        self.assertEqual(
            self.environment.thing.location, (-11, -11))

    def test_agent_attached_obj(self):
        self.assertEqual(
            self.environment.agent.attached_objects, [])


# class SwarmMultipleCarry(Agent):
#     """An minimalistic behavior tree for swarm agent implementing
#     MultipleCarry behavior
#     """
#     def __init__(self, name, model):
#         super().__init__(name, model)
#         self.location = ()

#         self.direction = model.random.rand() * (2 * np.pi)
#         # self.speed = 2
#         self.radius = 3

#         self.moveable = True
#         self.shared_content = dict()

#         root = py_trees.composites.Sequence("Sequence")

#         sense = NeighbourObjects('Sense')
#         sense.setup(0, self, 'Debris')

#         multiple_carry = CompositeMultipleCarry('MultipleCarry')
#         multiple_carry.setup(0, self, 'Debris')

#         move = Move('Move')
#         move.setup(0, self)

#         root.add_children([sense, multiple_carry, move])

#         self.behaviour_tree = py_trees.trees.BehaviourTree(root)

#         # Debugging stuffs for py_trees
#         # py_trees.logging.level = py_trees.logging.Level.DEBUG
#         # py_trees.display.print_ascii_tree(root)

#     def step(self):
#         self.behaviour_tree.tick()


# class MultipleCarryModel(Model):
#     """ A environment to model swarms """
#     def __init__(self, N, width, height, grid=10, seed=None):
#         if seed is None:
#             super(MultipleCarryModel, self).__init__(seed=None)
#         else:
#             super(MultipleCarryModel, self).__init__(seed)

#         self.num_agents = N

#         self.grid = Grid(width, height, grid)

#         self.schedule = SimultaneousActivation(self)

#         self.thing = Debris(id=1, location=(0, 0), radius=40)

#         self.grid.add_object_to_grid(self.thing.location, self.thing)

#         self.agent = []
#         for i in range(self.num_agents):
#             a = SwarmMultipleCarry(i, self)
#             self.schedule.add(a)
#             x = 1
#             y = 1
#             a.location = (x, y)
#             a.direction = 1.3561944901923448
#             self.grid.add_object_to_grid((x, y), a)
#             self.agent.append(a)

#     def step(self):
#         self.schedule.step()


# class TestMultipleCarry(TestCase):

#     def setUp(self):
#         self.environment = MultipleCarryModel(
#             2, 100, 100, 10, 123)

#         for i in range(25):
#             print(
#                 i, [agent.location for agent in self.environment.agent],
#                 self.environment.thing.location)
#             self.environment.step()

#     def tuple_round(self, loc):
#         loc1 = (np.round(loc[0]), np.round(loc[1]))
#         return loc1

#     def test_agent_loc(self):
#         # Check if the two agents end up at same location while carrying
#         # Heavy object
#         agent1_loc = self.tuple_round(self.environment.agent[0].location)
#         agent1_loc1 = (agent1_loc[0] + 1, agent1_loc[1] + 2)
#         agent2_loc = self.tuple_round(self.environment.agent[1].location)
#         self.assertEqual(agent1_loc1, agent2_loc)

#     def test_agent_object_loc(self):
#         # Check if the location of heavy object and one of the agent is
#         # almost same after moving
#         item_loc = self.tuple_round(self.environment.thing.location)
#         agent_loc = self.tuple_round(self.environment.agent[1].location)
#         self.assertEqual(item_loc, agent_loc)

#     def test_agent_move(self):
#         # Check if the item along with agents have moved to the border
#         # of the environment
#         item_loc = self.tuple_round(self.environment.thing.location)
#         self.assertEqual(item_loc, (49, 50))


# class SwarmSingleCarryFood(Agent):
#     """An minimalistic behavior tree for swarm agent implementing
#     CompositeCarry behavior
#     """
#     def __init__(self, name, model):
#         super().__init__(name, model)
#         self.location = ()

#         self.direction = model.random.rand() * (2 * np.pi)
#         # self.speed = 2
#         self.radius = 3

#         self.moveable = True
#         self.shared_content = dict()

#         root = py_trees.composites.Sequence("Sequence")
#         # Sensing the environemnt to find object to carry
#         lowest = NeighbourObjects('0')
#         lowest.setup(0, self, 'Food')

#         # First check if the item is carrable?
#         carryable = IsCarryable('SC_IsCarryable_1')
#         carryable.setup(0, self, 'Food')

#         # Then check if the item can be carried by a single agent
#         issinglecarry = IsSingleCarry('SC_IsSingleCarry_2')
#         issinglecarry.setup(0, self, 'Food')

#         # Finally, carry the object
#         singlecarry = SingleCarry('SC_SingleCarry_3')
#         singlecarry.setup(0, self, 'Food')

#         # Define a sequence to combine the primitive behavior
#         sc_sequence = py_trees.composites.Sequence('SC_SEQUENCE')
#         sc_sequence.add_children([carryable, issinglecarry, singlecarry])

#         # Creating composite single carry object
#         # singlecarry = CompositeSingleCarry('SingleCarry')
#         # singlecarry.setup(0, self, 'Food')

#         high = Explore('Explore')
#         high.setup(0, self)

#         root.add_children([lowest, sc_sequence, high])
#         self.behaviour_tree = py_trees.trees.BehaviourTree(root)

#         # Debugging stuffs for py_trees
#         # py_trees.logging.level = py_trees.logging.Level.DEBUG
#         # py_trees.display.print_ascii_tree(root)

#     def step(self):
#         self.behaviour_tree.tick()


# class SingleCarryFoodModel(Model):
#     """ A environment to model swarms """
#     def __init__(self, N, width, height, grid=10, seed=None):
#         if seed is None:
#             super(SingleCarryFoodModel, self).__init__(seed=None)
#         else:
#             super(SingleCarryFoodModel, self).__init__(seed)

#         self.num_agents = N

#         self.grid = Grid(width, height, grid)

#         self.schedule = SimultaneousActivation(self)

#         for a in range(5):
#             self.thing = Food(id=a, location=(0, 0), radius=16)

#             self.grid.add_object_to_grid(self.thing.location, self.thing)

#         for i in range(self.num_agents):
#             a = SwarmSingleCarryFood(i, self)
#             self.schedule.add(a)
#             x = 0
#             y = 0
#             a.location = (x, y)
#             a.direction = -2.3561944901923448
#             self.grid.add_object_to_grid((x, y), a)

#         self.agent = a

#     def step(self):
#         self.schedule.step()


# class TestSingleCarryFood(TestCase):

#     def setUp(self):
#         self.environment = SingleCarryFoodModel(
#             1, 100, 100, 10, 123)

#         for i in range(10):
#             self.environment.step()

#     def test_agent_path(self):
#         self.assertEqual(1, len(
#             self.environment.agent.attached_objects))
