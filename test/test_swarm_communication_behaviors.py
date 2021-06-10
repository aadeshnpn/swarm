"""Test files containging all the test cases for composit behaviors."""

from unittest import TestCase
from swarms.lib.agent import Agent
from swarms.lib.model import Model
from swarms.lib.time import SimultaneousActivation
from swarms.lib.space import Grid
from swarms.behaviors.scbehaviors import (
    Explore, CompositeSendSignal, CompositeReceiveSignal, CompositeDropPheromone, MoveAway, MoveTowards
    )
from swarms.behaviors.sbehaviors import (
    IsCarrying, NeighbourObjects, Move, IsCarryable,
    SingleCarry, IsSingleCarry
    )
from swarms.lib.objects import Obstacles, Pheromones, Sites, Debris, Food, Traps, Hub
from py_trees.trees import BehaviourTree
from py_trees.behaviour import Behaviour
from py_trees.composites import Sequence, Selector, Parallel
from py_trees import common, blackboard
from py_trees.decorators import FailureIsSuccess
import py_trees
import numpy as np


class SwarmSendSignal(Agent):
    """An minimalistic behavior tree for swarm agent implementing SendSignal
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

        self.blackboard = blackboard.Client(name=str(name))
        self.blackboard.register_key(key='neighbourobj', access=common.Access.WRITE)
        self.blackboard.neighbourobj = dict()

        if int(name) % 2 ==0:
            self.shared_content['Sites'] = {model.target}
            # Defining the composite behavior
            sendsig = CompositeSendSignal('SendSignal')
            # Setup for the behavior
            sendsig.setup(0, self, 'Sites')
        else:
            self.shared_content['Hub'] = {model.hub}
            # Defining the composite behavior
            sendsig = CompositeSendSignal('SendSignal')
            # Setup for the behavior
            sendsig.setup(0, self, 'Hub')

        # This behavior is just defined to check if the composite tree
        # can be combined with other primite behaviors
        seq = Sequence('Seq')
        exp = Explore('Explore')
        exp.setup(0, self, None)

        seq.add_children([sendsig, exp])

        # Since its root is a sequence, we can use it directly
        self.behaviour_tree = BehaviourTree(seq)

        # Debugging stuffs for py_trees
        # py_trees.logging.level = py_trees.logging.Level.DEBUG
        print(py_trees.display.ascii_tree(seq))

    def step(self):
        self.behaviour_tree.tick()


class SendSignalModel(Model):
    """ A environment to model swarms """
    def __init__(self, N, width, height, grid=10, seed=None):
        if seed is None:
            super(SendSignalModel, self).__init__(seed=None)
        else:
            super(SendSignalModel, self).__init__(seed)

        self.num_agents = N

        self.grid = Grid(width, height, grid)

        self.schedule = SimultaneousActivation(self)

        self.target = Sites(id=1, location=(45, 45), radius=5, q_value=0.5)
        self.grid.add_object_to_grid(self.target.location, self.target)

        self.hub = Hub(id=2, location=(-45, -45), radius=5)
        self.grid.add_object_to_grid(self.hub.location, self.hub)

        self.agents = []
        for i in range(self.num_agents):
            a = SwarmSendSignal(i, self)
            self.schedule.add(a)
            x = 0
            y = 0
            a.location = (x, y)
            a.direction = -2.3561944901923448
            self.grid.add_object_to_grid((x, y), a)
            self.agents += [a]

    def step(self):
        self.schedule.step()


class TestSendSignalSmallGrid(TestCase):

    def setUp(self):
        self.environment = SendSignalModel(2, 100, 100, 10, 123)

        for i in range(4):
            self.environment.step()

    def test_signal_match(self):
        # Checking if the agents reaches site or not
        # print(environment.agent.location, environment.agent.signals[0].location)
        _, glist = self.environment.grid.find_grid(self.environment.agents[0].location)
        sobjs = self.environment.grid.get_objects('Signal', glist)
        self.assertIn(self.environment.agents[0].signals[0], sobjs)
        self.assertIn(self.environment.agents[1].signals[0], sobjs)

    def test_communicated_objects(self):
        self.assertEqual(
            self.environment.target, self.environment.agents[0].signals[0].communicated_object)
        self.assertEqual(
            self.environment.hub, self.environment.agents[1].signals[0].communicated_object)


class SwarmReceiveSignal(Agent):
    """An minimalistic behavior tree for swarm agent implementing
       ReceiveSignal.
    """
    def __init__(self, name, model):
        super().__init__(name, model)
        self.location = ()

        self.direction = model.random.rand() * (2 * np.pi)
        # self.speed = 2
        self.radius = 3

        self.moveable = True
        self.shared_content = dict()

        self.blackboard = blackboard.Client(name=str(name))
        self.blackboard.register_key(
            key='neighbourobj', access=common.Access.WRITE)
        self.blackboard.neighbourobj = dict()

        if int(name) % 2 ==0:
            self.shared_content['Sites'] = {model.target}
            # Defining the composite behavior
            sendsig = CompositeSendSignal('SendSignal')
            # Setup for the behavior
            sendsig.setup(0, self, 'Sites')
        else:
            self.shared_content['Hub'] = {model.hub}
            # Defining the composite behavior
            sendsig = CompositeSendSignal('SendSignal')
            # Setup for the behavior
            sendsig.setup(0, self, 'Hub')

        # This behavior is just defined to check if the composite tree
        # can be combined with other primite behaviors
        seq = Sequence('Seq')

        recsignal = CompositeReceiveSignal('ReceiveSignal')
        recsignal.setup(0, self)

        exp = Explore('Explore')
        exp.setup(0, self, None)

        seq.add_children([exp, sendsig, recsignal])

        # Since its root is a sequence, we can use it directly
        self.behaviour_tree = BehaviourTree(seq)

        # Debugging stuffs for py_trees
        # if name ==1:
        #     py_trees.logging.level = py_trees.logging.Level.DEBUG
        print(py_trees.display.ascii_tree(seq))

    def step(self):
        self.behaviour_tree.tick()


class ReceiveSignalModel(Model):
    """ A environment to model swarms """
    def __init__(self, N, width, height, grid=10, seed=None):
        if seed is None:
            super(ReceiveSignalModel, self).__init__(seed=None)
        else:
            super(ReceiveSignalModel, self).__init__(seed)

        self.num_agents = N

        self.grid = Grid(width, height, grid)

        self.schedule = SimultaneousActivation(self)

        self.target = Sites(id=1, location=(45, 45), radius=5, q_value=0.5)
        self.grid.add_object_to_grid(self.target.location, self.target)

        self.hub = Hub(id=2, location=(-45, -45), radius=5)
        self.grid.add_object_to_grid(self.hub.location, self.hub)

        self.agents = []
        for i in range(self.num_agents):
            a = SwarmReceiveSignal(i, self)
            self.schedule.add(a)
            x = 0
            y = 0
            a.location = (x, y)
            a.direction = -2.3561944901923448
            self.grid.add_object_to_grid((x, y), a)
            self.agents += [a]

    def step(self):
        self.schedule.step()


class TestReceiveSignalSmallGrid(TestCase):

    def setUp(self):
        self.environment = ReceiveSignalModel(2, 100, 100, 10, 123)

        for i in range(10):
            self.environment.step()

    def test_signal_match(self):
        # Checking if the agents reaches site or not
        # print(environment.agent.location, environment.agent.signals[0].location)
        _, glist = self.environment.grid.find_grid(self.environment.agents[0].location)
        sobjs = self.environment.grid.get_objects('Signal', glist)
        self.assertIn(self.environment.agents[0].signals[0], sobjs)
        self.assertIn(self.environment.agents[1].signals[0], sobjs)

    def test_communicated_objects(self):
        self.assertEqual(self.environment.target, self.environment.agents[0].signals[0].communicated_object)
        self.assertEqual(self.environment.hub, self.environment.agents[1].signals[0].communicated_object)

    def test_shared_contents(self):
        self.assertEqual(self.environment.target, self.environment.agents[1].shared_content['Sites'].pop())
        self.assertEqual(self.environment.hub, self.environment.agents[0].shared_content['Hub'].pop())

    def test_shared_contents_objs(self):
        self.assertEqual(2, len(self.environment.agents[0].shared_content.keys()))
        self.assertEqual(2, len(self.environment.agents[1].shared_content.keys()))


class SwarmDropPheromone(Agent):
    """An minimalistic behavior tree for swarm agent implementing
       SwarmDropPheromone.
    """
    def __init__(self, name, model):
        super().__init__(name, model)
        self.location = ()

        self.direction = model.random.rand() * (2 * np.pi)
        # self.speed = 2
        self.radius = 3

        self.moveable = True
        self.shared_content = dict()

        self.blackboard = blackboard.Client(name=str(name))
        self.blackboard.register_key(
            key='neighbourobj', access=common.Access.WRITE)
        self.blackboard.neighbourobj = dict()

        self.shared_content['Sites'] = {model.target}
        self.shared_content['Hub'] = {model.hub}
        # Defining the composite behavior
        movewoards = MoveTowards('MoveTowards')
        movewoards.setup(0, self, 'Sites')

        dropphero = CompositeDropPheromone('DropPheromone')
        # Setup for the behavior
        dropphero.setup(0, self, None)

        # This behavior is just defined to check if the composite tree
        # can be combined with other primite behaviors
        seq = Sequence('Seq')

        seq.add_children([movewoards, dropphero])

        # Since its root is a sequence, we can use it directly
        self.behaviour_tree = BehaviourTree(seq)

        # Debugging stuffs for py_trees
        # py_trees.logging.level = py_trees.logging.Level.DEBUG
        print(py_trees.display.ascii_tree(seq))

    def step(self):
        self.behaviour_tree.tick()


class DropPheromoneModel(Model):
    """ A environment to model swarms """
    def __init__(self, N, width, height, grid=10, seed=None):
        if seed is None:
            super(DropPheromoneModel, self).__init__(seed=None)
        else:
            super(DropPheromoneModel, self).__init__(seed)

        self.num_agents = N

        self.grid = Grid(width, height, grid)

        self.schedule = SimultaneousActivation(self)

        self.target = Sites(id=1, location=(45, 45), radius=5, q_value=0.5)
        self.grid.add_object_to_grid(self.target.location, self.target)

        self.hub = Hub(id=2, location=(-45, -45), radius=5)
        self.grid.add_object_to_grid(self.hub.location, self.hub)

        # Since pheromones is central to the model.
        self.blackboard = blackboard.Client(name='Pheromones')
        self.blackboard.register_key(key='pheromones', access=common.Access.WRITE)
        self.blackboard.pheromones = list()

        self.agents = []
        for i in range(self.num_agents):
            a = SwarmDropPheromone(i, self)
            self.schedule.add(a)
            x = 0
            y = 0
            a.location = (x, y)
            a.direction = -2.3561944901923448
            self.grid.add_object_to_grid((x, y), a)
            self.agents += [a]

    def update_pheromones(self):
        for pheromone in self.blackboard.pheromones:
            pheromone.step()
            if pheromone.strength[pheromone.current_time] <= 0.0000:
                self.grid.remove_object_from_grid(pheromone.location, pheromone)
                self.blackboard.pheromones.remove(pheromone)

    def step(self):
        self.schedule.step()
        # self.update_pheromones()
        # Update the pheromones as well


class TestDropPheromoneSmallGrid(TestCase):

    def setUp(self):
        self.environment = DropPheromoneModel(1, 100, 100, 10, 123)

        for i in range(50):
            print(self.environment.agents[0].location)
            self.environment.step()

    def test_total_pheromone_dropped(self):
        self.assertEqual(5, len(self.environment.blackboard.pheromones))

    def test_agent_reached_site(self):
        self.assertEqual((40, 40), self.environment.agents[0].location)

    def test_agent_pheromone_direction(self):
        self.assertEqual(self.environment.agents[0].direction, self.environment.blackboard.pheromones[0].direction)



class DropPheromoneDecayModel(Model):
    """ A environment to model swarms """
    def __init__(self, N, width, height, grid=10, seed=None):
        if seed is None:
            super(DropPheromoneDecayModel, self).__init__(seed=None)
        else:
            super(DropPheromoneDecayModel, self).__init__(seed)

        self.num_agents = N

        self.grid = Grid(width, height, grid)

        self.schedule = SimultaneousActivation(self)

        self.target = Sites(id=1, location=(45, 45), radius=5, q_value=0.5)
        self.grid.add_object_to_grid(self.target.location, self.target)

        self.hub = Hub(id=2, location=(-45, -45), radius=5)
        self.grid.add_object_to_grid(self.hub.location, self.hub)

        # Since pheromones is central to the model.
        self.blackboard = blackboard.Client(name='Pheromones')
        self.blackboard.register_key(key='pheromones', access=common.Access.WRITE)
        self.blackboard.pheromones = list()

        self.agents = []
        for i in range(self.num_agents):
            a = SwarmDropPheromone(i, self)
            self.schedule.add(a)
            x = 0
            y = 0
            a.location = (x, y)
            a.direction = -2.3561944901923448
            self.grid.add_object_to_grid((x, y), a)
            self.agents += [a]

    def update_pheromones(self):
        for pheromone in self.blackboard.pheromones:
            pheromone.step()
            if pheromone.strength[pheromone.current_time] <= 0.0000:
                self.grid.remove_object_from_grid(pheromone.location, pheromone)
                self.blackboard.pheromones.remove(pheromone)

    def step(self):
        self.schedule.step()
        self.update_pheromones()
        # Update the pheromones as well


class TestDropPheromoneDecaySmallGrid(TestCase):

    def setUp(self):
        self.environment = DropPheromoneDecayModel(1, 100, 100, 10, 123)

        for i in range(50):
            # print(self.environment.agents[0].location)
            self.environment.step()

    def test_total_pheromone_dropped(self):
        self.assertEqual(1, len(self.environment.blackboard.pheromones))

    def test_agent_reached_site(self):
        self.assertEqual((40, 40), self.environment.agents[0].location)

    def test_agent_pheromone_direction(self):
        self.assertEqual(self.environment.agents[0].direction, self.environment.blackboard.pheromones[0].direction)

    def test_agent_pheromone_decay_time(self):
        self.assertEqual(10, self.environment.blackboard.pheromones[0].current_time)


# def main():
#     environment = DropPheromoneDecayModel(1, 100, 100, 10, 123)

#     for i in range(50):
#         agents = environment.agents
#         print(agents[0].location, [(p.location,p.direction, p.current_time) for p in environment.blackboard.pheromones])
#         environment.step()

# main()