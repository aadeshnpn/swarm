"""Test files containging all the test cases for composit behaviors."""

from unittest import TestCase
from swarms.lib.agent import Agent
from swarms.lib.model import Model
from swarms.lib.time import SimultaneousActivation
from swarms.lib.space import Grid
from swarms.behaviors.scbehaviors import (
    CompositeSensePheromone, Explore, CompositeSendSignal, CompositeReceiveSignal, CompositeDropPheromone, MoveAway, MoveTowards
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


class SwarmDropPheromoneDead(Agent):
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

        seq.add_children([dropphero, movewoards])

        # Since its root is a sequence, we can use it directly
        self.behaviour_tree = BehaviourTree(seq)

        # Debugging stuffs for py_trees
        # py_trees.logging.level = py_trees.logging.Level.DEBUG
        print(py_trees.display.ascii_tree(seq))

    def step(self):
        self.behaviour_tree.tick()


class DropPheromoneDeadModel(Model):
    """ A environment to model swarms """
    def __init__(self, N, width, height, grid=10, seed=None):
        if seed is None:
            super(DropPheromoneDeadModel, self).__init__(seed=None)
        else:
            super(DropPheromoneDeadModel, self).__init__(seed)

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
            a = SwarmDropPheromoneDead(i, self)
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


class TestDropPheromoneDropSmallGrid(TestCase):

    def setUp(self):
        self.environment = DropPheromoneDeadModel(1, 100, 100, 10, 123)

        for i in range(25):
            if i == 10:
                self.environment.agents[0].dead = True
            self.environment.step()

    def test_total_pheromone_dropped(self):
        self.assertEqual(1, len(self.environment.blackboard.pheromones))

    def test_agent_reached_site(self):
        self.assertEqual((10, 10), self.environment.agents[0].location)

    def test_agent_pheromone_direction(self):
        self.assertEqual(self.environment.agents[0].direction, self.environment.blackboard.pheromones[0].direction)

    def test_agent_pheromone_decay_time(self):
        self.assertEqual(14, self.environment.blackboard.pheromones[0].current_time)

    def test_agent_pheromone_repulsive(self):
        self.assertEqual(self.environment.blackboard.pheromones[0].attractive, False)



class SwarmSensePheromone(Agent):
    """An minimalistic behavior tree for swarm agent implementing
       SwarmSensePheromone.
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

        self.shared_content['Hub'] = {model.hub}
        # Defining the composite behavior
        movetwoards = MoveTowards('MoveTowards')
        movetwoards.setup(0, self, 'Hub')

        senseph = CompositeSensePheromone('SensePheromone')
        # Setup for the behavior
        senseph.setup(0, self, None)

        # This behavior is just defined to check if the composite tree
        # can be combined with other primite behaviors
        seq = Selector('Selector')

        seq.add_children([senseph, movetwoards])

        # Since its root is a sequence, we can use it directly
        self.behaviour_tree = BehaviourTree(seq)

        # Debugging stuffs for py_trees
        # py_trees.logging.level = py_trees.logging.Level.DEBUG
        print(py_trees.display.ascii_tree(seq))

    def step(self):
        self.behaviour_tree.tick()


class SensePheromoneModel(Model):
    """ A environment to model swarms """
    def __init__(self, N, width, height, grid=10, seed=None):
        if seed is None:
            super(SensePheromoneModel, self).__init__(seed=None)
        else:
            super(SensePheromoneModel, self).__init__(seed)

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
            if i % 2 == 0:
                a = SwarmDropPheromone(i, self)
                x = -30
                y = -30
            else:
                a = SwarmSensePheromone(i, self)
                x = 0
                y = 0
            self.schedule.add(a)

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


class TestDropPheromoneSmallGrid(TestCase):

    def setUp(self):
        self.environment = SensePheromoneModel(2, 100, 100, 10, 123)

        for i in range(50):
            print(self.environment.agents[0].location)
            self.environment.step()

    def test_total_pheromone_dropped(self):
        self.assertEqual(1, len(self.environment.blackboard.pheromones))

    def test_agent_reached_site(self):
        self.assertEqual((35, 35), self.environment.agents[0].location)
        self.assertEqual((36, 36), self.environment.agents[1].location)

    def test_agent_pheromone_direction(self):
        self.assertEqual(self.environment.agents[0].direction, self.environment.blackboard.pheromones[0].direction)


# def main():
#     environment = SensePheromoneModel(2, 100, 100, 10, 123)

#     for i in range(50):
#         agents = environment.agents
#         print(i, [(p.location, p.direction, p.current_time, p.attractive) for p in environment.blackboard.pheromones])
#         print([a.location for a in agents])
#         environment.step()

# main()