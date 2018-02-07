from unittest import TestCase
from lib.agent import Agent
from lib.model import Model
from lib.time import SimultaneousActivation
from lib.space import Grid
from swarms.sbehaviors import (
    IsCarryable, IsSingleCarry, SingleCarry,
    NeighbourObjects, IsMultipleCarry, IsInPartialAttached,
    InitiateMultipleCarry, IsEnoughStrengthToCarry
    )
from swarms.objects import Derbis
import py_trees
import numpy as np


# Class to tets series of carry and drop behaviors

class SwarmAgentSingleCarry(Agent):
    """ An minimalistic behavior tree for swarm agent implementing carry behavior"""
    def __init__(self, name, model):
        super().__init__(name, model)
        self.location = ()

        self.direction = model.random.rand() * (2 * np.pi)
        self.speed = 2
        self.radius = 3

        self.moveable = True
        self.shared_content = dict()
        
        root = py_trees.composites.Sequence("Sequence")
        lowest = NeighbourObjects('0')
        lowest.setup(0, self, 'Derbis')

        low = IsCarryable('1')
        low.setup(0, self, 'Derbis')
        medium = IsSingleCarry('2')
        medium.setup(0, self, 'Derbis')
        high = SingleCarry('3')
        high.setup(0, self, 'Derbis')

        root.add_children([lowest, low, medium, high])
        self.behaviour_tree = py_trees.trees.BehaviourTree(root)
        py_trees.display.print_ascii_tree(root)
        
    def step(self):
        self.behaviour_tree.tick()

    def advance(self):
        pass


class SwarmAgentMultipleCarry(Agent):
    """ An minimalistic behavior tree for swarm agent implementing multiple carry behavior"""
    def __init__(self, name, model):
        super().__init__(name, model)
        self.location = ()

        self.direction = model.random.rand() * (2 * np.pi)
        self.speed = 2
        self.radius = 3
        self.moveable = True
        self.shared_content = dict()
        
        root = py_trees.composites.Sequence("Sequence")
        lowest = NeighbourObjects('0')
        lowest.setup(0, self, 'Derbis')

        low = IsCarryable('1')
        low.setup(0, self, 'Derbis')

        medium = IsMultipleCarry('2')
        medium.setup(0, self, 'Derbis')

        r1Sequence = py_trees.composites.Sequence("R1Sequence")        
        r2Sequence = py_trees.composites.Sequence("R2Sequence")                
        r1Selector = py_trees.composites.Selector("R1Selector")

        high1 = IsInPartialAttached('3')
        high1.setup(0, self, 'Derbis')

        high2 = InitiateMultipleCarry('4')
        high2.setup(0, self, 'Derbis')        

        high3 = IsEnoughStrengthToCarry('5')
        high3.setup(0, self, 'Derbis')        

        r2Sequence.add_children([high3])        

        r1Selector.add_children([high1, high2])

        r1Sequence.add_children([medium, r1Selector, r2Sequence])

        root.add_children([lowest, low, r1Sequence])

        self.behaviour_tree = py_trees.trees.BehaviourTree(root)
        py_trees.logging.level = py_trees.logging.Level.DEBUG
        py_trees.display.print_ascii_tree(root)
        
    def step(self):
        self.behaviour_tree.tick()

    def advance(self):
        pass


class SingleCarrySwarmEnvironmentModel(Model):
    """ A environemnt to model swarms """
    def __init__(self, N, width, height, grid=10, seed=None):
        if seed is None:
            super(SingleCarrySwarmEnvironmentModel, self).__init__(seed=None)
        else:
            super(SingleCarrySwarmEnvironmentModel, self).__init__(seed)

        self.num_agents = N

        self.grid = Grid(width, height, grid)

        self.schedule = SimultaneousActivation(self)
        
        # self.target = Sites(id=1, location=(45, 45), radius=5, q_value=0.5)
        self.thing = Derbis(id=1, location=(0, 0), radius=4)
        self.grid.add_object_to_grid(self.thing.location, self.thing)

        for i in range(self.num_agents):
            a = SwarmAgentSingleCarry(i, self)
            self.schedule.add(a)
            x = 1
            y = 1
            a.location = (x, y)
            a.direction = -2.3561944901923448
            self.grid.add_object_to_grid((x, y), a)

        self.agent = a

    def step(self):
        self.schedule.step()


class MultipleCarrySwarmEnvironmentModel(Model):
    """ A environemnt to model swarms """
    def __init__(self, N, width, height, grid=10, seed=None):
        if seed is None:
            super(MultipleCarrySwarmEnvironmentModel, self).__init__(seed=None)
        else:
            super(MultipleCarrySwarmEnvironmentModel, self).__init__(seed)

        self.num_agents = N

        self.grid = Grid(width, height, grid)

        self.schedule = SimultaneousActivation(self)
        
        # self.target = Sites(id=1, location=(45, 45), radius=5, q_value=0.5)
        self.thing = Derbis(id=1, location=(0, 0), radius=38)
        self.grid.add_object_to_grid(self.thing.location, self.thing)

        for i in range(self.num_agents):
            a = SwarmAgentMultipleCarry(i, self)
            self.schedule.add(a)
            x = 1
            y = 1
            a.location = (x, y)
            a.direction = -2.3561944901923448
            self.grid.add_object_to_grid((x, y), a)

        self.agent = a

    def step(self):
        self.schedule.step()        


class TestSingleCarrySameLocationSwarmSmallGrid(TestCase):
    
    def setUp(self):
        self.environment = SingleCarrySwarmEnvironmentModel(1, 100, 100, 10, 123)

        for i in range(2):
            self.environment.step()

    def test_agent_path(self):
        self.assertEqual(self.environment.agent.attached_objects[0], self.environment.thing)


class TestMultipleCarrySameLocationSwarmSmallGrid(TestCase):
    
    def setUp(self):
        self.environment = MultipleCarrySwarmEnvironmentModel(2, 100, 100, 10, 123)

        for i in range(2):
            self.environment.step()

    def test_agent_path(self):
        self.assertEqual(2, 3)
