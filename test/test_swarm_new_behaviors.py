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
    AvoidTrapObstaclesBehaviour, NewMoveTowards, NewExplore,
    NewMoveAway, Explore
    )    
from swarms.lib.objects import Sites, Hub, Obstacles, Traps
import py_trees
from py_trees import Blackboard
import numpy as np


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

        self.blackboard = Blackboard()
        self.blackboard.shared_content = dict()

        self.shared_content[type(model.target).__name__] = {model.target}

        low = GoTo('1')
        low.setup(0, self, type(model.target).__name__)
        high = Move('2')
        high.setup(0, self)
        root.add_children([low, high])
        self.behaviour_tree = py_trees.trees.BehaviourTree(root)

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

        for i in range(25):
            self.environment.step()
            print(i, self.environment.agent.location)

    def test_agent_path(self):
        self.assertEqual(self.environment.agent.location, (-1, 7))


# Class to tets the avoid behavior for the agent
class SwarmAgentAvoid(Agent):
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
        self.carryable = False
        self.shared_content = dict()

        root = py_trees.composites.Sequence("Sequence")

        self.blackboard = Blackboard()
        self.blackboard.shared_content = dict()

        self.shared_content[type(model.target).__name__] = {model.target}

        low = GoTo('1')
        low.setup(0, self, type(model.target).__name__)
        # medium = NeighbourObjects('2')
        # medium.setup(0, self, item=None)        
        # med = AvoidSObjects('3')
        # med.setup(0, self, type(model.obstacle).__name__)       
        medium = AvoidTrapObstaclesBehaviour('2')
        medium.setup(0, self, None)        
        high = Move('4')
        high.setup(0, self)
        # root.add_children([low, medium, med, high])
        root.add_children([low, medium, high])        
        self.behaviour_tree = py_trees.trees.BehaviourTree(root)
        # py_trees.display.print_ascii_tree(root)
        # py_trees.logging.level = py_trees.logging.Level.DEBUG

    def step(self):
        self.behaviour_tree.tick()



class AvoidSwarmEnvironmentModel(Model):
    """ A environemnt to model swarms """
    def __init__(self, N, width, height, grid=10, seed=None):
        if seed is None:
            super(AvoidSwarmEnvironmentModel, self).__init__(seed=None)
        else:
            super(AvoidSwarmEnvironmentModel, self).__init__(seed)

        self.num_agents = N

        self.grid = Grid(width, height, grid)

        self.schedule = SimultaneousActivation(self)

        self.obstacle = Obstacles(id=2, location=(9, 9), radius=5)

        self.target = Sites(id=1, location=(45, 45), radius=5, q_value=0.5)

        self.grid.add_object_to_grid(self.target.location, self.target)
        self.grid.add_object_to_grid(self.obstacle.location, self.obstacle)

        for i in range(self.num_agents):
            a = SwarmAgentAvoid(i, self)
            self.schedule.add(a)
            x = -30
            y = -41
            a.location = (x, y)
            a.direction = -2.3561944901923448
            self.grid.add_object_to_grid((x, y), a)

        self.agent = a

    def step(self):
        self.schedule.step()



class TestAvoidSwarmSmallGrid(TestCase):

    def setUp(self):
        self.environment = AvoidSwarmEnvironmentModel(1, 100, 100, 10, 123)

        for i in range(120):
            self.environment.step()
            print(i, self.environment.agent.location)   

    def test_agent_path(self):
        self.assertEqual(self.environment.agent.location, (45, 45))


    # def test_agent_goal(self):
    #     site = self.environment.grid.get_objects(
    #         'Sites', self.environment.grid.find_grid(self.environment.agent.location)[1]
    #         )
    #     self.assertGreater(len(site), 0)
        

# Class to tets agent dead in trap
class SwarmAgentTrap(Agent):
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

        self.blackboard = Blackboard()
        self.blackboard.shared_content = dict()

        self.shared_content[type(model.target).__name__] = {model.target}

        low = GoTo('1')
        low.setup(0, self, type(model.target).__name__)
        high = Move('2')
        high.setup(0, self)
        root.add_children([low, high])
        self.behaviour_tree = py_trees.trees.BehaviourTree(root)

    def step(self):
        self.behaviour_tree.tick()



class TrapSwarmEnvironmentModel(Model):
    """ A environemnt to model swarms """
    def __init__(self, N, width, height, grid=10, seed=None):
        if seed is None:
            super(TrapSwarmEnvironmentModel, self).__init__(seed=None)
        else:
            super(TrapSwarmEnvironmentModel, self).__init__(seed)

        self.num_agents = N

        self.grid = Grid(width, height, grid)

        self.schedule = SimultaneousActivation(self)

        self.target = Traps(id=1, location=(20,20), radius=8)
        self.grid.add_object_to_grid(self.target.location, self.target)

        for i in range(self.num_agents):
            a = SwarmAgentTrap(i, self)
            self.schedule.add(a)
            x = -45
            y = -45
            a.location = (x, y)
            a.direction = -2.3561944901923448
            self.grid.add_object_to_grid((x, y), a)

        self.agent = a

    def step(self):
        self.schedule.step()



# class TestTrapSwarmSmallGrid(TestCase):

#     def setUp(self):
#         self.environment = TrapSwarmEnvironmentModel(1, 100, 100, 10, 123)

#         for i in range(50):
#             self.environment.step()
#             # print(i, self.environment.agent.location, self.environment.agent.dead)

#     def test_agent_path(self):
#         self.assertEqual(self.environment.agent.location, (22, 20))         

#     def test_agent_dead(self):
#         self.assertEqual(self.environment.agent.dead, True)                 



# Class to tets the avoid trap behavior for the agent
class SwarmAgentAvoidTrap(Agent):
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
        self.carryable = False
        self.shared_content = dict()

        root = py_trees.composites.Sequence("Sequence")

        self.blackboard = Blackboard()
        self.blackboard.shared_content = dict()

        self.shared_content[type(model.target).__name__] = {model.target}

        low = GoTo('1')
        low.setup(0, self, type(model.target).__name__)
        
        # medium = NeighbourObjects('2')
        # medium.setup(0, self, item=None)        
        # med = AvoidSObjects('3')
        # med.setup(0, self, type(model.trap).__name__)       
        
        medium = AvoidTrapObstaclesBehaviour('2')
        medium.setup(0, self)

        high = Move('4')
        high.setup(0, self)
        # root.add_children([low, medium, med, high])
        root.add_children([low, medium, high])        
        self.behaviour_tree = py_trees.trees.BehaviourTree(root)
        # py_trees.display.print_ascii_tree(root)
        # py_trees.logging.level = py_trees.logging.Level.DEBUG

    def step(self):
        self.behaviour_tree.tick()



class AvoidTrapSwarmEnvironmentModel(Model):
    """ A environemnt to model swarms """
    def __init__(self, N, width, height, grid=10, seed=None):
        if seed is None:
            super(AvoidTrapSwarmEnvironmentModel, self).__init__(seed=None)
        else:
            super(AvoidTrapSwarmEnvironmentModel, self).__init__(seed)

        self.num_agents = N

        self.grid = Grid(width, height, grid)

        self.schedule = SimultaneousActivation(self)

        self.trap = Traps(id=2, location=(9, 9), radius=5)

        self.target = Sites(id=1, location=(45, 45), radius=5, q_value=0.5)

        self.grid.add_object_to_grid(self.target.location, self.target)
        self.grid.add_object_to_grid(self.trap.location, self.trap)

        for i in range(self.num_agents):
            a = SwarmAgentAvoidTrap(i, self)
            self.schedule.add(a)
            x = -45
            y = -45
            a.location = (x, y)
            a.direction = -2.3561944901923448
            self.grid.add_object_to_grid((x, y), a)

        self.agent = a

    def step(self):
        self.schedule.step()



# class TestAvoidTrapSwarmSmallGrid(TestCase):

#     def setUp(self):
#         self.environment = AvoidTrapSwarmEnvironmentModel(1, 100, 100, 10, 123)

#         for i in range(120):
#             self.environment.step()
#             print(i, self.environment.agent.location, self.environment.agent.dead)

#     def test_agent_path(self):
#         self.assertEqual(self.environment.agent.location, (45, 45))


#     def test_agent_goal(self):
#         site = self.environment.grid.get_objects(
#             'Sites', self.environment.grid.find_grid(self.environment.agent.location)[1]
#             )
#         self.assertGreater(len(site), 0)
        


# Class to tets the avoid trap behavior for the agent
class SwarmAgentAvoidTrapNew(Agent):
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
        self.carryable = False
        self.shared_content = dict()

        root = py_trees.composites.Sequence("Sequence")

        self.blackboard = Blackboard()
        self.blackboard.shared_content = dict()

        self.shared_content[type(model.target).__name__] = {model.target}

        low = NewMoveTowards('1')
        low.setup(0, self, type(model.target).__name__)

        # root.add_children([low, medium, med, high])
        root.add_children([low])        
        self.behaviour_tree = py_trees.trees.BehaviourTree(root)
        # py_trees.display.print_ascii_tree(root)
        # py_trees.logging.level = py_trees.logging.Level.DEBUG

    def step(self):
        self.behaviour_tree.tick()



class AvoidTrapNewSwarmEnvironmentModel(Model):
    """ A environemnt to model swarms """
    def __init__(self, N, width, height, grid=10, seed=None):
        if seed is None:
            super(AvoidTrapNewSwarmEnvironmentModel, self).__init__(seed=None)
        else:
            super(AvoidTrapNewSwarmEnvironmentModel, self).__init__(seed)

        self.num_agents = N

        self.grid = Grid(width, height, grid)

        self.schedule = SimultaneousActivation(self)

        self.trap = Traps(id=2, location=(9, 9), radius=5)

        self.target = Sites(id=1, location=(45, 45), radius=5, q_value=0.5)

        self.grid.add_object_to_grid(self.target.location, self.target)
        self.grid.add_object_to_grid(self.trap.location, self.trap)

        for i in range(self.num_agents):
            a = SwarmAgentAvoidTrapNew(i, self)
            self.schedule.add(a)
            x = -45
            y = -45
            a.location = (x, y)
            a.direction = -2.3561944901923448
            self.grid.add_object_to_grid((x, y), a)

        self.agent = a

    def step(self):
        self.schedule.step()



# class TestAvoidTrapNewSwarmSmallGrid(TestCase):

#     def setUp(self):
#         self.environment = AvoidTrapNewSwarmEnvironmentModel(1, 100, 100, 10, 123)

#         for i in range(120):
#             self.environment.step()
#             # print(i, self.environment.agent.location, self.environment.agent.dead)

#     def test_agent_path(self):
#         self.assertEqual(self.environment.agent.location, (45, 45))


#     def test_agent_goal(self):
#         site = self.environment.grid.get_objects(
#             'Sites', self.environment.grid.find_grid(self.environment.agent.location)[1]
#             )
#         self.assertGreater(len(site), 0)
                


# Class to tets the avoid trap behavior for the agent
class SwarmAgentExploreNew(Agent):
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
        self.carryable = False
        self.shared_content = dict()

        root = py_trees.composites.Sequence("Sequence")

        self.blackboard = Blackboard()
        self.blackboard.shared_content = dict()

        self.shared_content[type(model.target).__name__] = {model.target}

        low = NewExplore('1')
        low.setup(0, self)

        # root.add_children([low, medium, med, high])
        root.add_children([low])        
        self.behaviour_tree = py_trees.trees.BehaviourTree(root)
        # py_trees.display.print_ascii_tree(root)
        # py_trees.logging.level = py_trees.logging.Level.DEBUG

    def step(self):
        self.behaviour_tree.tick()



class ExploreNewSwarmEnvironmentModel(Model):
    """ A environemnt to model swarms """
    def __init__(self, N, width, height, grid=10, seed=None):
        if seed is None:
            super(ExploreNewSwarmEnvironmentModel, self).__init__(seed=None)
        else:
            super(ExploreNewSwarmEnvironmentModel, self).__init__(seed)

        self.num_agents = N

        self.grid = Grid(width, height, grid)

        self.schedule = SimultaneousActivation(self)

        self.obstacle = Obstacles(id=2, location=(9, 9), radius=5)

        self.target = Sites(id=1, location=(45, 45), radius=5, q_value=0.5)

        self.grid.add_object_to_grid(self.target.location, self.target)
        self.grid.add_object_to_grid(self.obstacle.location, self.obstacle)

        for i in range(self.num_agents):
            a = SwarmAgentExploreNew(i, self)
            self.schedule.add(a)
            x = 25
            y = 25
            a.location = (x, y)
            a.direction = -2.3561944901923448
            self.grid.add_object_to_grid((x, y), a)

        self.agent = a

    def step(self):
        self.schedule.step()



# class TestExploreNewSwarmSmallGrid(TestCase):

#     def setUp(self):
#         self.environment = ExploreNewSwarmEnvironmentModel(1, 100, 100, 10, 123)

#         for i in range(50):
#             self.environment.step()
#             # print(i, self.environment.agent.location)

#     def test_agent_path(self):
#         self.assertEqual(self.environment.agent.location, (-9, 18))


# Class to tets the avoid trap behavior for the agent
class SwarmAgentMoveAway(Agent):
    """ An minimalistic behavior tree for swarm agent implementing 
    move away behavior.
    """
    def __init__(self, name, model):
        super().__init__(name, model)
        self.location = ()

        self.direction = model.random.rand() * (2 * np.pi)
        self.speed = 2
        self.radius = 5
        self.moveable = True
        self.carryable = False
        self.shared_content = dict()

        root = py_trees.composites.Sequence("Sequence")

        self.blackboard = Blackboard()
        self.blackboard.shared_content = dict()

        self.shared_content[type(model.target).__name__] = {model.target}

        low = NewMoveAway('1')
        # low = NewMoveTowards('1')        
        low.setup(0, self, type(model.target).__name__)

        # root.add_children([low, medium, med, high])
        root.add_children([low])        
        self.behaviour_tree = py_trees.trees.BehaviourTree(root)
        # py_trees.display.print_ascii_tree(root)
        # py_trees.logging.level = py_trees.logging.Level.DEBUG

    def step(self):
        self.behaviour_tree.tick()



class MoveAwaySwarmEnvironmentModel(Model):
    """ A environemnt to model swarms """
    def __init__(self, N, width, height, grid=10, seed=None):
        if seed is None:
            super(MoveAwaySwarmEnvironmentModel, self).__init__(seed=None)
        else:
            super(MoveAwaySwarmEnvironmentModel, self).__init__(seed)

        self.num_agents = N

        self.grid = Grid(width, height, grid)

        self.schedule = SimultaneousActivation(self)

        self.obstacle = Obstacles(id=2, location=(9, 9), radius=5)

        self.target = Sites(id=1, location=(45, 45), radius=5, q_value=0.5)

        self.grid.add_object_to_grid(self.target.location, self.target)
        self.grid.add_object_to_grid(self.obstacle.location, self.obstacle)

        for i in range(self.num_agents):
            a = SwarmAgentMoveAway(i, self)
            self.schedule.add(a)
            x = 35
            y = 35
            a.location = (x, y)
            a.direction = -2.3561944901923448
            self.grid.add_object_to_grid((x, y), a)

        self.agent = a

    def step(self):
        self.schedule.step()



# class TestMoveAwaySwarmSmallGrid(TestCase):

#     def setUp(self):
#         self.environment = MoveAwaySwarmEnvironmentModel(1, 100, 100, 10, 123)

#         for i in range(50):
#             self.environment.step()
#             # print(i, self.environment.agent.location, self.environment.agent.dead)

#     def test_agent_path(self):
#         self.assertEqual(self.environment.agent.location, (-14, -20))

                
class SwarmAgentExplore(Agent):
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
        self.carryable = False
        self.shared_content = dict()

        root = py_trees.composites.Sequence("Sequence")

        self.blackboard = Blackboard()
        self.blackboard.shared_content = dict()

        self.shared_content[type(model.target).__name__] = {model.target}

        low = Explore('1')
        low.setup(0, self)

        # root.add_children([low, medium, med, high])
        root.add_children([low])        
        self.behaviour_tree = py_trees.trees.BehaviourTree(root)
        # py_trees.display.print_ascii_tree(root)
        # py_trees.logging.level = py_trees.logging.Level.DEBUG

    def step(self):
        self.behaviour_tree.tick()



class ExploreSwarmEnvironmentModel(Model):
    """ A environemnt to model swarms """
    def __init__(self, N, width, height, grid=10, seed=None):
        if seed is None:
            super(ExploreSwarmEnvironmentModel, self).__init__(seed=None)
        else:
            super(ExploreSwarmEnvironmentModel, self).__init__(seed)

        self.num_agents = N

        self.grid = Grid(width, height, grid)

        self.schedule = SimultaneousActivation(self)

        self.obstacle = Obstacles(id=2, location=(9, 9), radius=15)

        self.target = Sites(id=1, location=(45, 45), radius=5, q_value=0.5)

        self.grid.add_object_to_grid(self.target.location, self.target)
        self.grid.add_object_to_grid(self.obstacle.location, self.obstacle)

        for i in range(self.num_agents):
            a = SwarmAgentExplore(i, self)
            self.schedule.add(a)
            x = -35
            y = -35
            a.location = (x, y)
            a.direction = -2.3561944901923448
            self.grid.add_object_to_grid((x, y), a)

        self.agent = a

    def step(self):
        self.schedule.step()



# class TestExploreSwarmSmallGrid(TestCase):

#     def setUp(self):
#         self.environment = ExploreSwarmEnvironmentModel(1, 100, 100, 10, 123)

#         for i in range(100):
#             self.environment.step()
#             # print(i, self.environment.agent.location)

#     def test_agent_path(self):
#         self.assertEqual(self.environment.agent.location, (15, -11))

