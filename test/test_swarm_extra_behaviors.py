from unittest import TestCase
from lib.agent import Agent
from lib.model import Model
from lib.time import SimultaneousActivation
from lib.space import Grid
from swarms.sbehaviors import (
    IsCarryable, IsSingleCarry, SingleCarry,
    NeighbourObjects, IsMultipleCarry, IsInPartialAttached,
    InitiateMultipleCarry, IsEnoughStrengthToCarry,
    Move, GoTo, Drop, IsDropable, IsCarrying
    )
from swarms.objects import Derbis, Sites, Hub, Food
import py_trees
from py_trees.composites import RepeatUntilFalse
import numpy as np


# Class to test accleration and velocity models
class SwarmAgentGoTo(Agent):
    """ An minimalistic behavior tree for swarm agent implementing goto 
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
        # Vairables related to motion
        self.accleration = [0, 0]
        self.velocity = [0, 0]

        root = py_trees.composites.Sequence("Sequence")
        low = GoTo('1')
        low.setup(0, self, 'Sites')
        high = Move('2')
        high.setup(0, self)

        higest = py_trees.meta.inverter(NeighbourObjects)('3')
        higest.setup(0, self, 'Sites')
        root.add_children([higest, low, high])
        self.behaviour_tree = py_trees.trees.BehaviourTree(root)
        #py_trees.logging.level = py_trees.logging.Level.DEBUG                
        py_trees.display.print_ascii_tree(root)

    def step(self):
        self.behaviour_tree.tick() 


class GoToSwarmEnvironmentModel(Model):
    """ A environment to model swarms """
    def __init__(self, N, width, height, grid=10, seed=None):
        if seed is None:
            super(GoToSwarmEnvironmentModel, self).__init__(seed=None)
        else:
            super(GoToSwarmEnvironmentModel, self).__init__(seed)

        self.num_agents = N

        self.grid = Grid(width, height, grid)

        self.schedule = SimultaneousActivation(self)
        
        self.target = Sites(id=1, location=(45, 45), radius=5, q_value=0.5)
        self.grid.add_object_to_grid(self.target.location, self.target)

        for i in range(self.num_agents):
            a = SwarmAgentGoTo(i, self)
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
        self.environment = GoToSwarmEnvironmentModel(1, 100, 100, 10, 123)

        for i in range(69):
            print(self.environment.agent.location)
            self.environment.step()

    def test_agent_path(self):
        self.assertEqual(self.environment.agent.location, (40, 40))


# Class to test series of carry and drop behaviors
class SwarmAgentSingleCarry(Agent):
    """ An minimalistic behavior tree for swarm agent
    implementing carry behavior
    """
    def __init__(self, name, model):
        super().__init__(name, model)
        self.location = ()

        self.direction = model.random.rand() * (2 * np.pi)
        self.speed = 2
        self.radius = 3

        self.moveable = True
        self.shared_content = dict()
        py_trees.logging.level = py_trees.logging.Level.DEBUG        
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


class SingleCarrySwarmEnvironmentModel(Model):
    """ A environment to model swarms """
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
            x = 0
            y = 0
            a.location = (x, y)
            a.direction = -2.3561944901923448
            self.grid.add_object_to_grid((x, y), a)

        self.agent = a

    def step(self):
        self.schedule.step()


class TestSingleCarrySameLocationSwarmSmallGrid(TestCase):
    
    def setUp(self):
        self.environment = SingleCarrySwarmEnvironmentModel(
            1, 100, 100, 10, 123)

        for i in range(1):
            self.environment.step()

    def test_agent_path(self):
        self.assertEqual(
            self.environment.agent.attached_objects[0], self.environment.thing)


class SwarmAgentSingleCarryDrop(Agent):
    """ An minimalistic behavior tree for swarm agent
    implementing carry and drop behavior
    """
    def __init__(self, name, model):
        super().__init__(name, model)
        self.location = ()

        self.direction = model.random.rand() * (2 * np.pi)
        self.speed = 2
        self.radius = 3

        self.moveable = True
        self.shared_content = dict()        
        
        #self.shared_content['Hub'] = model.hub
        #self.shared_content['Sites'] = model.site

        root = py_trees.composites.Sequence("Sequence")
        mseq = py_trees.composites.Sequence('MSequence')
        select = py_trees.composites.Selector('RSelector')
        carryseq = py_trees.composites.Sequence('CSequence')
        
        lowest = NeighbourObjects('0')
        lowest.setup(0, self, 'Food')

        low = IsCarryable('1')
        low.setup(0, self, 'Food')

        medium = IsSingleCarry('2')
        medium.setup(0, self, 'Food')

        high = SingleCarry('3')
        high.setup(0, self, 'Food')        

        carryseq.add_children([lowest, low, medium, high])

        repeathub = RepeatUntilFalse("RepeatSeq")
        
        high1 = py_trees.meta.inverter(NeighbourObjects)('4')
        #high1 = NeighbourObjects('4')
        high1.setup(0, self, 'Hub')   

        med1 = py_trees.meta.inverter(GoTo)('5')     
        #med1 = GoTo('5')
        med1.setup(0, self, 'Hub')

        #low1 = py_trees.meta.inverter(Move)('6')
        low1 = Move('6')
        low1.setup(0, self, None)


        # Drop
        dropseq = py_trees.composites.Sequence('DSequence')
        c1 = IsCarrying('7')
        c1.setup(0, self, 'Food')

        d1 = IsDropable('8')
        d1.setup(0, self, 'Hub')

        d2 = Drop('9')
        d2.setup(0, self, 'Food')

        dropseq.add_children([c1, d1, d2])

        repeathub.add_children([high1, med1, low1])
        mseq.add_children([carryseq, repeathub])
        select.add_children([dropseq, mseq])        

        root.add_children([select])

        self.behaviour_tree = py_trees.trees.BehaviourTree(root)

        py_trees.logging.level = py_trees.logging.Level.DEBUG        
        py_trees.display.print_ascii_tree(root)

    def step(self):
        self.behaviour_tree.tick()

    def advance(self):
        pass


class SingleCarryDropSwarmEnvironmentModel(Model):
    """ A environment to model swarms """
    def __init__(self, N, width, height, grid=10, seed=None):
        if seed is None:
            super(SingleCarryDropSwarmEnvironmentModel, self).__init__(seed=None)
        else:
            super(SingleCarryDropSwarmEnvironmentModel, self).__init__(seed)

        self.num_agents = N

        self.grid = Grid(width, height, grid)

        self.schedule = SimultaneousActivation(self)
        
        self.hub = Hub(id=2, location=(0, 0), radius=5)
        self.grid.add_object_to_grid(self.hub.location, self.hub)
        self.site = Sites(id=3, location=(40, 40), radius=20)
        self.grid.add_object_to_grid(self.site.location, self.site)

        for i in range(self.num_agents):
            a = SwarmAgentSingleCarryDrop(i, self)
            self.schedule.add(a)
            x = 40
            y = 40
            a.location = (x, y)
            a.direction = -2.3561944901923448
            self.grid.add_object_to_grid((x, y), a)

        self.agent = a

        for i in range(self.num_agents*2):
            f = Food(i, location=(40, 40), radius=2)
            self.grid.add_object_to_grid(f.location, f)

    def step(self):
        self.schedule.step()


class TestSingleCarryDropSwarmSmallGrid(TestCase):
    
    def setUp(self):
        self.environment = SingleCarryDropSwarmEnvironmentModel(
            1, 100, 100, 10, 123)

        for i in range(10):
            self.environment.step()

    def test_agent_food(self):
        # Testing after the food has been transported to hub and dropped. Is the location of the food dropped
        # and the agent is same or not.
        transported_food = self.environment.grid.get_objects_from_grid('Food', self.environment.agent.location)[0]
        self.assertEqual (self.environment.agent.location, transported_food.location)

    def test_agent_reach_hub(self):
        # Testing is the agent has reached near to the hub. The agent won't exactly land on the hub
        self.assertEqual(self.environment.agent.location, (8, 8))        

    def test_agent_drop(self):
        # Testing if the food has been dropped or not. The agent attached_objects should be empty is this case
        self.assertEqual([],self.environment.agent.attached_objects)


class SwarmAgentSingleCarryDropReturn(Agent):
    """ An minimalistic behavior tree for swarm agent
    implementing carry and drop behavior
    """
    def __init__(self, name, model):
        super().__init__(name, model)
        self.location = ()

        self.direction = model.random.rand() * (2 * np.pi)
        self.speed = 2
        self.radius = 3

        self.moveable = True
        self.shared_content = dict()        
        
        #self.shared_content['Hub'] = model.hub
        #self.shared_content['Sites'] = model.site

        #root = py_trees.composites.Sequence("Sequence")
        #root = py_trees.composites.Selector('Selector')
        mseq = py_trees.composites.Sequence('MSequence')
        nseq = py_trees.composites.Sequence('NSequence')
        select = py_trees.composites.Selector('RSelector')
        carryseq = py_trees.composites.Sequence('CSequence')
        dropseq = py_trees.composites.Sequence('DSequence')
        
        lowest = NeighbourObjects('0')
        lowest.setup(0, self, 'Food')

        low = IsCarryable('1')
        low.setup(0, self, 'Food')

        medium = IsSingleCarry('2')
        medium.setup(0, self, 'Food')

        high = SingleCarry('3')
        high.setup(0, self, 'Food')        

        carryseq.add_children([lowest, low, medium, high])

        repeathub = RepeatUntilFalse("RepeatSeqHub")
        repeatsite = RepeatUntilFalse("RepeatSeqSite")
        
        high1 = py_trees.meta.inverter(NeighbourObjects)('4')
        #high1 = NeighbourObjects('4')
        high1.setup(0, self, 'Hub')   

        med1 = py_trees.meta.inverter(GoTo)('5')     
        #med1 = GoTo('5')
        med1.setup(0, self, 'Hub')

        #low1 = py_trees.meta.inverter(Move)('6')
        low1 = Move('6')
        low1.setup(0, self, None)

        high2 = py_trees.meta.inverter(NeighbourObjects)('12')
        #high2 = NeighbourObjects('12')
        high2.setup(0, self, 'Sites')   

        #med2 = py_trees.meta.inverter(GoTo)('13')     
        med2 = GoTo('13')
        med2.setup(0, self, 'Sites')

        #low1 = py_trees.meta.inverter(Move)('6')
        low2 = Move('14')
        low2.setup(0, self, None)

        # Drop
        dropseq = py_trees.composites.Sequence('DSequence')
        c1 = IsCarrying('7')
        c1.setup(0, self, 'Food')

        d1 = IsDropable('8')
        d1.setup(0, self, 'Hub')

        d2 = Drop('9')
        d2.setup(0, self, 'Food')

        dropseq.add_children([c1, d1, d2])

        repeathub.add_children([high1, med1, low1])
        repeatsite.add_children([high2, med2, low2])

        mseq.add_children([carryseq, repeathub])
        nseq.add_children([dropseq, repeatsite])

        select.add_children([nseq, mseq])
        self.behaviour_tree = py_trees.trees.BehaviourTree(select)

        py_trees.logging.level = py_trees.logging.Level.DEBUG        
        py_trees.display.print_ascii_tree(select)
        self.shared_content['Sites'] = [model.site]

    def step(self):
        self.behaviour_tree.tick()

    def advance(self):
        pass


class SingleCarryDropReturnSwarmEnvironmentModel(Model):
    """ A environment to model swarms """
    def __init__(self, N, width, height, grid=10, seed=None):
        if seed is None:
            super(SingleCarryDropReturnSwarmEnvironmentModel, self).__init__(seed=None)
        else:
            super(SingleCarryDropReturnSwarmEnvironmentModel, self).__init__(seed)

        self.num_agents = N

        self.grid = Grid(width, height, grid)

        self.schedule = SimultaneousActivation(self)
        
        self.hub = Hub(id=2, location=(0, 0), radius=5)
        self.grid.add_object_to_grid(self.hub.location, self.hub)
        self.site = Sites(id=3, location=(40, 40), radius=5)
        self.grid.add_object_to_grid(self.site.location, self.site)
        
        for i in range(self.num_agents):
            f = Food(i, location=(40, 40), radius=2)
            self.grid.add_object_to_grid(f.location, f)
            self.food = f

        for i in range(self.num_agents):
            a = SwarmAgentSingleCarryDropReturn(i, self)
            self.schedule.add(a)
            x = 40
            y = 40
            a.location = (x, y)
            a.direction = -2.3561944901923448
            self.grid.add_object_to_grid((x, y), a)

        self.agent = a

    def step(self):
        self.schedule.step()


class TestSingleCarryDropReturnSwarmSmallGrid(TestCase):

    def setUp(self):
        self.environment = SingleCarryDropReturnSwarmEnvironmentModel(
            1, 100, 100, 10, 123)

    def test_agent_food(self):
        # Testing after the food has been transported to hub and dropped. Is the location of the food dropped
        # and the hub same
        for i in range(2):
            self.environment.step()

        transported_food = self.environment.grid.get_objects_from_grid('Food', self.environment.hub.location)[0]
        self.assertEqual(self.environment.food, transported_food)

    def test_agent_reach_hub(self):
        # Testing is the agent has reached near to the hub. The agent won't exactly land on the hub
        for i in range(1):
            self.environment.step()
        self.assertEqual(self.environment.agent.location, (8, 8))        

    def test_agent_drop(self):
        # Testing if the food has been dropped or not. The agent attached_objects should be empty is this case
        for i in range(2):
            self.environment.step()
        self.assertEqual([], self.environment.agent.attached_objects)  

    def test_agent_reach_site(self):
        # Testing if the agent has reached site after dropping food in hub
        for i in range(2):
            self.environment.step()
        self.assertEqual(self.environment.agent.location, self.environment.site.location)


class SwarmAgentMultipleCarry(Agent):
    # An minimalistic behavior tree for swarm agent
    #implementing multiple carry behavior
    #
    def __init__(self, name, model):
        super().__init__(name, model)
        self.location = ()

        self.direction = model.random.rand() * (2 * np.pi)
        self.speed = 2
        self.radius = 3
        self.moveable = True
        self.shared_content = dict()

        # Variables realted to carry
        self.used_capacity = 0

        # Vairables related to motion
        self.accleration = [0, 0]
        self.velocity = [0, 0]

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

        high4 = Move('6')
        high4.setup(0, self)

        r2Sequence.add_children([high3, high4])        

        r1Selector.add_children([high1, high2])

        r1Sequence.add_children([medium, r1Selector, r2Sequence])

        root.add_children([lowest, low, r1Sequence])

        self.behaviour_tree = py_trees.trees.BehaviourTree(root)
        #py_trees.logging.level = py_trees.logging.Level.DEBUG
        #py_trees.display.print_ascii_tree(root)

    def unused_capacity(self):
        return self.capacity - self.used_capacity

    def step(self):
        self.behaviour_tree.tick()

    def advance(self):
        pass


class MultipleCarrySwarmEnvironmentModel(Model):
    # A environment to model swarms
    def __init__(self, N, width, height, grid=10, seed=None):
        if seed is None:
            super(MultipleCarrySwarmEnvironmentModel, self).__init__(seed=None)
        else:
            super(MultipleCarrySwarmEnvironmentModel, self).__init__(seed)

        self.num_agents = N

        self.grid = Grid(width, height, grid)

        self.schedule = SimultaneousActivation(self)
        
        # self.target = Sites(id=1, location=(45, 45), radius=5, q_value=0.5)
        self.thing = Derbis(id=1, location=(0, 0), radius=40)
        self.grid.add_object_to_grid(self.thing.location, self.thing)

        self.agent = []
        for i in range(self.num_agents):
            a = SwarmAgentMultipleCarry(i, self)
            self.schedule.add(a)
            x = 1
            y = 1
            a.location = (x, y)
            a.direction = -2.3561944901923448
            self.grid.add_object_to_grid((x, y), a)

            self.agent.append(a)

    def step(self):
        self.schedule.step() 


class TestMultipleCarrySameLocationSwarmSmallGrid(TestCase):
    
    def setUp(self):
        self.environment = MultipleCarrySwarmEnvironmentModel(
            2, 100, 100, 10, 123)
        print('starting locatio', self.environment.thing.location)
        for i in range(2):
            self.environment.step()
            print('movement of thing', self.environment.thing.location)            

    def test_agent_path(self):
        self.assertEqual(2, 3)

"""
class TestCoolMultipleCarryFunction(TestCase):

    def setUp(self):
        self.environment = MultipleCarrySwarmEnvironmentModel(
            4, 100, 100, 10, 123)

        for agent in self.environment.agent:
            agent.used_capacity = self.environment.random.randint(0, 8)

    def test_distribute_weight(self):
        agents = [(
            agent.name, agent.capacity - agent.used_capacity
            ) for agent in self.environment.agent]
        
        print(agents)
        for i in range(1):
            self.environment.step()
        self.assertEqual(5, 4)


"""
"""
# Behavior defined to move using accleration. It doesn't 
# work that well in this context
class Move(Behaviour):
    def __init__(self, name):
        super(Move, self).__init__(name)

    def setup(self, timeout, agent):
        self.agent = agent
        self.dt = 1.0 / 30

    def initialise(self):
        pass

    def update_partial_attached_objects(self):
        pass
        for item in self.agent.partial_attached_objects:
            accleration = self.agent.force / item.weight
            print(self.agent.force, item.weight)
            velocity = accleration * self.dt
            direction = self.agent.direction
            x = int(item.location[0] + np.cos(direction) * velocity)
            y = int(item.location[1] + np.sin(direction) * velocity)
            object_agent = list(item.agents.keys())[0]
            new_location, direction = object_agent.model.grid.check_limits(
                (x, y), direction)
            print('update', item.location, new_location)
            object_agent.model.grid.move_object(
                item.location, item, new_location)
            item.location = new_location

    def update(self):
        accleration = self.agent.force / self.agent.get_weight()
        self.agent.accleration[0] += np.cos(self.agent.direction) * accleration
        self.agent.accleration[1] += np.sin(self.agent.direction) * accleration
        if self.agent.accleration[0] >= 20:
            self.agent.accleration[0] = 20
        if self.agent.accleration[1] >= 20:
            self.agent.accleration[1] = 20

        self.agent.velocity[0] += self.agent.accleration[0] * self.dt
        self.agent.velocity[1] += self.agent.accleration[1] * self.dt
        x = round(self.agent.location[0] + self.agent.velocity[0] * self.dt)
        y = round(self.agent.location[0] + self.agent.velocity[1] * self.dt)

        new_location, direction = self.agent.model.grid.check_limits(
            (x, y), self.agent.direction)
        self.agent.model.grid.move_object(
            self.agent.location, self.agent, new_location)
        self.agent.location = new_location
        self.agent.direction = direction

        print('move', self.agent.name, self.agent.accleration, self.agent.velocity, self.agent.location, self.agent.direction)
        # Full carried object moves along the agent
        for item in self.agent.attached_objects:
            item.location = self.agent.location

        # Partially carried object is moved by the team effor
        self.update_partial_attached_objects()

        return Status.SUCCESS
"""        