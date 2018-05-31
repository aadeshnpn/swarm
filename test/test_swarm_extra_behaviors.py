from unittest import TestCase
from swarms.lib.agent import Agent
from swarms.lib.model import Model
from swarms.lib.time import SimultaneousActivation
from swarms.lib.space import Grid
from swarms.sbehaviors import (
    IsCarryable, IsSingleCarry, SingleCarry,
    NeighbourObjects, IsMultipleCarry, IsInPartialAttached,
    InitiateMultipleCarry, IsEnoughStrengthToCarry,
    Move, GoTo, Drop, IsDropable, IsCarrying, Towards, DropPartial,
    IsVisitedBefore, RandomWalk, SignalDoesNotExists, SendSignal,
    ReceiveSignal
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
        # py_trees.logging.level = py_trees.logging.Level.DEBUG
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
        # high1 = NeighbourObjects('4')
        high1.setup(0, self, 'Hub')

        med1 = py_trees.meta.inverter(GoTo)('5')
        # med1 = GoTo('5')
        med1.setup(0, self, 'Hub')

        # low1 = py_trees.meta.inverter(Move)('6')
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
            super(SingleCarryDropSwarmEnvironmentModel, self).__init__(
                seed=None)
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

        for i in range(self.num_agents * 2):
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
        # Testing after the food has been transported to hub and dropped. Is
        # the location of the food dropped
        # and the agent is same or not.
        transported_food = self.environment.grid.get_objects_from_grid(
            'Food', self.environment.agent.location)[0]
        self.assertEqual(
            self.environment.agent.location, transported_food.location)

    def test_agent_reach_hub(self):
        # Testing is the agent has reached near to the hub. The agent won't
        # exactly land on the hub
        self.assertEqual(self.environment.agent.location, (8, 8))

    def test_agent_drop(self):
        # Testing if the food has been dropped or not. The agent
        # attached_objects should be empty is this case
        self.assertEqual([], self.environment.agent.attached_objects)


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

        # self.shared_content['Hub'] = model.hub
        # self.shared_content['Sites'] = model.site

        # root = py_trees.composites.Sequence("Sequence")
        # root = py_trees.composites.Selector('Selector')
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
        # high1 = NeighbourObjects('4')
        high1.setup(0, self, 'Hub')

        med1 = py_trees.meta.inverter(GoTo)('5')
        # med1 = GoTo('5')
        med1.setup(0, self, 'Hub')

        # low1 = py_trees.meta.inverter(Move)('6')
        low1 = Move('6')
        low1.setup(0, self, None)

        high2 = py_trees.meta.inverter(NeighbourObjects)('12')
        # high2 = NeighbourObjects('12')
        high2.setup(0, self, 'Sites')

        # med2 = py_trees.meta.inverter(GoTo)('13')
        med2 = GoTo('13')
        med2.setup(0, self, 'Sites')

        # low1 = py_trees.meta.inverter(Move)('6')
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
            super(SingleCarryDropReturnSwarmEnvironmentModel, self).__init__(
                seed=None)
        else:
            super(SingleCarryDropReturnSwarmEnvironmentModel, self).__init__(
                seed)

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
        # Testing after the food has been transported to hub and dropped. Is
        # the location of the food dropped
        # and the hub same
        for i in range(2):
            self.environment.step()

        transported_food = self.environment.grid.get_objects_from_grid(
            'Food', self.environment.hub.location)[0]
        self.assertEqual(self.environment.food, transported_food)

    def test_agent_reach_hub(self):
        # Testing is the agent has reached near to the hub. The agent
        # won't exactly land on the hub
        for i in range(1):
            self.environment.step()
        self.assertEqual(self.environment.agent.location, (8, 8))

    def test_agent_drop(self):
        # Testing if the food has been dropped or not. The agent
        # attached_objects should be empty is this case
        for i in range(2):
            self.environment.step()
        self.assertEqual([], self.environment.agent.attached_objects)

    def test_agent_reach_site(self):
        # Testing if the agent has reached site after dropping food in hub
        for i in range(2):
            self.environment.step()
        self.assertEqual(
            self.environment.agent.location, self.environment.site.location)


class SwarmAgentMultipleCarry(Agent):
    # An minimalistic behavior tree for swarm agent
    # implementing multiple carry behavior
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
        # py_trees.logging.level = py_trees.logging.Level.DEBUG
        # py_trees.display.print_ascii_tree(root)

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

        for i in range(60):
            self.environment.step()

    def tuple_round(self, loc):
        loc1 = (np.round(loc[0]), np.round(loc[1]))
        return loc1

    def test_agent_loc(self):
        # Check if the two agents end up at same location while carrying
        # Heavy object
        agent1_loc = self.tuple_round(self.environment.agent[0].location)
        agent2_loc = self.tuple_round(self.environment.agent[1].location)
        self.assertEqual(agent1_loc, agent2_loc)

    def test_agent_object_loc(self):
        # Check if the location of heavy object and one of the agent is
        # almost same after moving
        item_loc = self.tuple_round(self.environment.thing.location)
        agent_loc = self.tuple_round(self.environment.agent[0].location)
        self.assertEqual(item_loc, agent_loc)


class SwarmAgentMultipleCarryDrop(Agent):
    # An minimalistic behavior tree for swarm agent
    # implementing multiple carry behavior and drop
    # in the hub
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

        self.shared_content['Hub'] = [model.hub]

        carryroot = py_trees.composites.Sequence("Sequence")
        lowest = NeighbourObjects('0')
        lowest.setup(0, self, 'Derbis')

        low = IsCarryable('1')
        low.setup(0, self, 'Derbis')

        medium = IsMultipleCarry('2')
        medium.setup(0, self, 'Derbis')

        r1Sequence = py_trees.composites.Sequence("R1Sequence")
        r1Selector = py_trees.composites.Selector("R1Selector")
        mainSelector = py_trees.composites.Selector("MainSelector")

        high1 = IsInPartialAttached('3')
        high1.setup(0, self, 'Derbis')

        high2 = InitiateMultipleCarry('4')
        high2.setup(0, self, 'Derbis')

        high3 = IsEnoughStrengthToCarry('5')
        high3.setup(0, self, 'Derbis')

        high4 = GoTo('6')
        high4.setup(0, self, 'Hub')

        high5 = Move('7')
        high5.setup(0, self)

        nearHub1 = NeighbourObjects('13')
        nearHub1.setup(0, self, 'Hub')

        r1Selector.add_children([high1, high2])
        r1Sequence.add_children([medium, r1Selector])

        carryroot.add_children([lowest, low, r1Sequence])
        mainSelector.add_children([nearHub1, carryroot])

        # Adding new sub-tree for drop logic for multiple carry
        droproot = py_trees.composites.Selector("DropSelector")
        moveSequence = py_trees.composites.Sequence("MoveSequence")
        dropSequence = py_trees.composites.Sequence("DropSequence")

        nearHub = py_trees.meta.inverter(NeighbourObjects)('8')
        nearHub.setup(0, self, 'Hub')

        moveSequence.add_children([nearHub, high3, high4, high5])

        high6 = NeighbourObjects('9')
        high6.setup(0, self, 'Hub')

        high7 = IsDropable('10')
        high7.setup(0, self, 'Hub')

        high8 = IsInPartialAttached('11')
        high8.setup(0, self, 'Derbis')

        high9 = DropPartial('12')
        high9.setup(0, self, 'Derbis')

        dropSequence.add_children([high6, high7, high8, high9])
        droproot.add_children([moveSequence, dropSequence])

        root = py_trees.composites.Sequence("Root")
        root.add_children([mainSelector, droproot])

        self.behaviour_tree = py_trees.trees.BehaviourTree(root)
        # py_trees.logging.level = py_trees.logging.Level.DEBUG
        # py_trees.display.print_ascii_tree(root)

    def unused_capacity(self):
        return self.capacity - self.used_capacity

    def step(self):
        self.behaviour_tree.tick()

    def advance(self):
        pass


class MultipleCarrySwarmDropEnvironmentModel(Model):
    # A environment to model swarms
    def __init__(self, N, width, height, grid=10, seed=None):
        if seed is None:
            super(MultipleCarrySwarmDropEnvironmentModel, self).__init__(
                seed=None)
        else:
            super(MultipleCarrySwarmDropEnvironmentModel, self).__init__(seed)

        self.num_agents = N

        self.grid = Grid(width, height, grid)

        self.schedule = SimultaneousActivation(self)

        self.hub = Hub(id=2, location=(42, 42), radius=5)
        self.grid.add_object_to_grid(self.hub.location, self.hub)

        self.thing = Derbis(id=1, location=(-10, -10), radius=32)
        self.grid.add_object_to_grid(self.thing.location, self.thing)

        self.agent = []
        for i in range(self.num_agents):
            a = SwarmAgentMultipleCarryDrop(i, self)
            self.schedule.add(a)
            x = -10
            y = -10
            a.location = (x, y)
            a.direction = -2.3561944901923448
            self.grid.add_object_to_grid((x, y), a)

            self.agent.append(a)

    def step(self):
        self.schedule.step()


class TestMultipleCarryDropSameLocationSwarmSmallGrid(TestCase):

    def setUp(self):
        self.environment = MultipleCarrySwarmDropEnvironmentModel(
            2, 100, 100, 10, 123)

        for i in range(28):
            self.environment.step()

    def tuple_round(self, loc):
        loc1 = (np.round(loc[0]), np.round(loc[1]))
        return loc1

    def test_agent_loc(self):
        # Check if the two agents end up at same location while carrying
        # Heavy object and dropping off in the hub
        agent2_loc = self.tuple_round(self.environment.agent[1].location)
        derbis_loc = self.tuple_round(self.environment.thing.location)
        self.assertEqual(agent2_loc, derbis_loc)

    def test_derbis_drop(self):
        # Check if the object has been dropped at hub. For this just see,
        # is the partial attached portion of agent is empty.
        # Check if the agents dict in the object is empty
        agent1_attached = self.environment.agent[0].partial_attached_objects
        agent2_attached = self.environment.agent[1].partial_attached_objects
        item_attached = self.environment.thing.agents

        self.assertEqual(agent1_attached, agent2_attached)
        self.assertEqual(dict(), item_attached)


class SwarmAgentRandomSingleCarryDropReturn(Agent):
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

        # self.shared_content['Hub'] = model.hub
        # self.shared_content['Sites'] = model.site
        self.shared_content['Hub'] = [model.hub]
        # root = py_trees.composites.Sequence("Sequence")
        # root = py_trees.composites.Selector('Selector')
        mseq = py_trees.composites.Sequence('MSequence')
        nseq = py_trees.composites.Sequence('NSequence')
        select = py_trees.composites.Selector('RSelector')
        carryseq = py_trees.composites.Sequence('CSequence')
        dropseq = py_trees.composites.Sequence('DSequence')

        lowest1 = py_trees.meta.inverter(NeighbourObjects)('00')
        lowest1.setup(0, self, 'Hub')

        lowest11 = NeighbourObjects('0')
        lowest11.setup(0, self, 'Sites')

        lowest = NeighbourObjects('0')
        lowest.setup(0, self, 'Food')

        low = IsCarryable('1')
        low.setup(0, self, 'Food')

        medium = IsSingleCarry('2')
        medium.setup(0, self, 'Food')

        high = SingleCarry('3')
        high.setup(0, self, 'Food')

        carryseq.add_children([lowest1, lowest11, lowest, low, medium, high])

        repeathub = RepeatUntilFalse("RepeatSeqHub")
        repeatsite = RepeatUntilFalse("RepeatSeqSite")

        high1 = py_trees.meta.inverter(NeighbourObjects)('4')
        # high1 = NeighbourObjects('4')
        high1.setup(0, self, 'Hub')

        med1 = GoTo('5')
        med1.setup(0, self, 'Hub')

        # low1 = py_trees.meta.inverter(Move)('6')
        low1 = Move('6')
        low1.setup(0, self, None)

        high2 = py_trees.meta.inverter(NeighbourObjects)('12')
        # high2 = NeighbourObjects('12')
        high2.setup(0, self, 'Sites')

        # med2 = py_trees.meta.inverter(GoTo)('13')
        med2 = GoTo('13')
        med2.setup(0, self, 'Sites')

        # low1 = py_trees.meta.inverter(Move)('6')
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

        # For randomwalk to work the agents shouldn't know the location of Site
        v1 = py_trees.meta.inverter(IsVisitedBefore)('15')
        v1.setup(0, self, 'Sites')

        r1 = RandomWalk('16')
        r1.setup(0, self, None)

        m1 = Move('17')
        m1.setup(0, self, None)

        randseq = py_trees.composites.Sequence('RSequence')
        randseq.add_children([v1, r1, m1])

        select.add_children([nseq, mseq, randseq])

        self.behaviour_tree = py_trees.trees.BehaviourTree(select)

    def step(self):
        self.behaviour_tree.tick()

    def advance(self):
        pass


class RandomSingleCarryDropReturnSwarmModel(Model):
    """A environment to model swarms."""

    def __init__(self, N, width, height, grid=10, seed=None):
        if seed is None:
            super(RandomSingleCarryDropReturnSwarmModel, self).__init__(
                seed=None)
        else:
            super(RandomSingleCarryDropReturnSwarmModel, self).__init__(
                seed)

        self.num_agents = N

        self.grid = Grid(width, height, grid)

        self.schedule = SimultaneousActivation(self)

        self.hub = Hub(id=2, location=(0, 0), radius=5)
        self.grid.add_object_to_grid(self.hub.location, self.hub)
        self.site = Sites(id=3, location=(-35, -5), radius=10)
        self.grid.add_object_to_grid(self.site.location, self.site)

        for i in range(self.num_agents * 5):
            f = Food(i, location=(-35, -5), radius=2)
            self.grid.add_object_to_grid(f.location, f)

        for i in range(self.num_agents):
            a = SwarmAgentRandomSingleCarryDropReturn(i, self)
            self.schedule.add(a)
            x = 0
            y = 0
            a.location = (x, y)
            a.direction = -2.3561944901923448
            self.grid.add_object_to_grid((x, y), a)

        self.agent = a

    def step(self):
        self.schedule.step()


class TestRandomSingleCarryDropReturnSwarmSmallGrid(TestCase):

    def setUp(self):
        self.environment = RandomSingleCarryDropReturnSwarmModel(
            1, 100, 100, 10, 123456)

    def test_agent_food(self):
        # Testing after the food has been transported to hub and dropped. Is
        # the total number of food is transported to hub
        for i in range(40):
            self.environment.step()
        transported_food = self.environment.grid.get_objects_from_grid(
            'Food', self.environment.hub.location)
        self.assertEqual(5, len(transported_food))


class SwarmAgentRandomWalk(Agent):
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

        self.shared_content['Hub'] = [model.hub]

        # Just checking the randomwalk behavior with many agent
        # how much they explore the environment

        r1 = RandomWalk('1')
        r1.setup(0, self, None)

        m1 = Move('2')
        m1.setup(0, self, None)

        randseq = py_trees.composites.Sequence('RSequence')
        randseq.add_children([r1, m1])
        self.behaviour_tree = py_trees.trees.BehaviourTree(randseq)

    def step(self):
        self.behaviour_tree.tick()

    def advance(self):
        pass


class RandomWalkSwarmEnvironmentModel(Model):
    """ A environment to model swarms """
    def __init__(self, N, width, height, grid=10, seed=None):
        if seed is None:
            super(RandomWalkSwarmEnvironmentModel, self).__init__(
                seed=None)
        else:
            super(RandomWalkSwarmEnvironmentModel, self).__init__(
                seed)

        self.num_agents = N

        self.grid = Grid(width, height, grid)

        self.schedule = SimultaneousActivation(self)

        self.hub = Hub(id=2, location=(0, 0), radius=5)
        self.grid.add_object_to_grid(self.hub.location, self.hub)

        self.agents = []
        for i in range(self.num_agents):
            a = SwarmAgentRandomWalk(i, self)
            self.schedule.add(a)
            x = 0
            y = 0
            a.location = (x, y)
            a.direction = -2.3561944901923448
            self.grid.add_object_to_grid((x, y), a)
            self.agents.append(a)

    def step(self):
        self.schedule.step()


class TestRandomWalkSwarmSmallGrid(TestCase):

    def setUp(self):
        self.environment = RandomWalkSwarmEnvironmentModel(
            50, 100, 100, 10, 123)

    def test_agent_exploration(self):
        # Testing is the agents cover the whole environment by randomly
        # move over it
        grid_visited = set()
        for i in range(130):
            self.environment.step()
            for a in self.environment.agents:
                _, grid_val = self.environment.grid.find_grid(a.location)
                grid_visited.add(grid_val)
        self.assertEqual(100, len(grid_visited))


class SwarmSignal(Agent):
    """ An minimalistic behavior tree for swarm agent
    implementing signal behavior
    """
    def __init__(self, name, model):
        super().__init__(name, model)
        self.location = ()

        self.direction = model.random.rand() * (2 * np.pi)
        self.speed = 2
        self.radius = 3

        self.moveable = True
        self.shared_content = dict()

        self.shared_content['Hub'] = [model.hub]
        self.shared_content['Food'] = [model.food]
        # Just checking the Signal behaviors using the behaviors defined
        # below which moves the agent towards the hub.
        n1 = py_trees.meta.inverter(NeighbourObjects)('1')
        n1.setup(0, self, 'Hub')

        g1 = GoTo('2')
        g1.setup(0, self, 'Hub')

        m1 = Move('3')
        m1.setup(0, self, None)

        sense = py_trees.composites.Sequence('Sequence')
        sense.add_children([n1, g1, m1])

        s1 = SignalDoesNotExists('4')
        s1.setup(0, self, 'Food')

        s2 = SendSignal('5')
        s2.setup(0, self, 'Food')

        signal = py_trees.composites.Sequence('SequenceSignal')
        signal.add_children([s1, s2])

        select = py_trees.composites.Selector('Selector')
        select.add_children([signal, sense])

        self.behaviour_tree = py_trees.trees.BehaviourTree(select)

        py_trees.logging.level = py_trees.logging.Level.DEBUG
        py_trees.display.print_ascii_tree(select)

    def step(self):
        self.behaviour_tree.tick()

    def advance(self):
        pass


class SignalModel(Model):
    """ A environment to model swarms """
    def __init__(self, N, width, height, grid=10, seed=None):
        if seed is None:
            super(SignalModel, self).__init__(
                seed=None)
        else:
            super(SignalModel, self).__init__(
                seed)

        self.num_agents = N

        self.grid = Grid(width, height, grid)

        self.schedule = SimultaneousActivation(self)

        self.hub = Hub(id=2, location=(0, 0), radius=5)
        self.grid.add_object_to_grid(self.hub.location, self.hub)

        self.food = Food(id=9, location=(-45, -45), radius=3)
        self.grid.add_object_to_grid(self.food.location, self.food)

        self.agents = []
        for i in range(self.num_agents):
            a = SwarmSignal(i, self)
            self.schedule.add(a)
            x = 45
            y = 45
            a.location = (x, y)
            a.direction = -2.3561944901923448
            self.grid.add_object_to_grid((x, y), a)
            self.agents.append(a)

    def step(self):
        self.schedule.step()


class TestSignalSwarmSmallGrid(TestCase):
    # Testing the functionality of signal. Signal needs to
    # broadcast information and move along with the agent. The
    # broadcasted information is about any object. Signal object
    # attaches the object for information transfer

    def setUp(self):
        self.environment = SignalModel(
            1, 100, 100, 10, 123)

    def test_agent_signal_movement(self):

        for i in range(20):
            self.environment.step()

        agent = self.environment.agents[0]
        agent_loc = agent.location
        _, grid_val = self.environment.grid.find_grid(agent_loc)
        signal = self.environment.grid.get_objects('Signal', grid_val)

        # Checking is the signal is moving along with the agent
        self.assertEqual(signal, agent.signals)

    def test_agent_signal_send(self):

        for i in range(20):
            self.environment.step()

        agent = self.environment.agents[0]

        # Checking if the agent has reached the hub
        self.assertEqual(agent.location, (9, 9))


class SwarmSignalRec1(Agent):
    """ An minimalistic behavior tree for swarm agent
    implementing signal behavior with both send and receive
    """
    def __init__(self, name, model):
        super().__init__(name, model)
        self.location = ()

        self.direction = model.random.rand() * (2 * np.pi)
        self.speed = 2
        self.radius = 3

        self.moveable = True
        self.shared_content = dict()

        # Just checking the Signal behaviors using the behaviors defined
        # below which moves the agent towards the site after receiving signal
        # from other agents about a site.
        n1 = py_trees.meta.inverter(NeighbourObjects)('1')
        n1.setup(0, self, 'Sites')

        g1 = GoTo('1')
        g1.setup(0, self, 'Sites')

        m1 = Move('3')
        m1.setup(0, self, None)

        sense = py_trees.composites.Sequence('Sequence')
        sense.add_children([n1, g1, m1])

        r1 = ReceiveSignal('4')
        r1.setup(0, self, 'Signal')

        # s2 = SendSignal('5')
        # s2.setup(0, self, 'Food')

        signal = py_trees.composites.Selector('Signal')
        signal.add_children([r1])

        select = py_trees.composites.Selector('Selector')
        select.add_children([signal, sense])

        self.behaviour_tree = py_trees.trees.BehaviourTree(select)

        # py_trees.logging.level = py_trees.logging.Level.DEBUG
        # py_trees.display.print_ascii_tree(select)

    def step(self):
        self.behaviour_tree.tick()

    def advance(self):
        pass


class SwarmSignalRec2(Agent):
    """ An minimalistic behavior tree for swarm agent
    implementing signal behavior
    """
    def __init__(self, name, model):
        super().__init__(name, model)
        self.location = ()

        self.direction = model.random.rand() * (2 * np.pi)
        self.speed = 2
        self.radius = 3

        self.moveable = True
        self.shared_content = dict()

        self.shared_content['Hub'] = [model.hub]
        # Just checking the Signal behaviors using the behaviors defined
        # below which moves the agent towards the hub. When its finds
        # sites, its starts to signal
        n1 = py_trees.meta.inverter(NeighbourObjects)('1')
        n1.setup(0, self, 'Hub')

        g1 = GoTo('2')
        g1.setup(0, self, 'Hub')

        m1 = Move('3')
        m1.setup(0, self, None)

        sense = py_trees.composites.Sequence('Sequence')
        sense.add_children([n1, g1, m1])

        s1 = SignalDoesNotExists('4')
        s1.setup(0, self, 'Sites')

        s2 = SendSignal('5')
        s2.setup(0, self, 'Sites')

        s3 = NeighbourObjects('6')
        s3.setup(0, self, 'Sites')

        signal = py_trees.composites.Sequence('SequenceSignal')
        signal.add_children([s3, s1, s2])

        select = py_trees.composites.Selector('Selector')
        select.add_children([signal, sense])

        self.behaviour_tree = py_trees.trees.BehaviourTree(select)

        # py_trees.logging.level = py_trees.logging.Level.DEBUG
        # py_trees.display.print_ascii_tree(select)

    def step(self):
        self.behaviour_tree.tick()

    def advance(self):
        pass


class SignalModelRec(Model):
    """ A environment to model swarms """
    def __init__(self, N, width, height, grid=10, seed=None):
        if seed is None:
            super(SignalModelRec, self).__init__(
                seed=None)
        else:
            super(SignalModelRec, self).__init__(
                seed)

        self.num_agents = N

        self.grid = Grid(width, height, grid)

        self.schedule = SimultaneousActivation(self)

        self.hub = Hub(id=2, location=(0, 0), radius=5)
        self.grid.add_object_to_grid(self.hub.location, self.hub)

        self.site = Sites(id=5, location=(20, 20), radius=5)
        self.grid.add_object_to_grid(self.site.location, self.site)

        self.agents = []

        # Define agent which is going to send the signal
        for i in range(self.num_agents):
            a = SwarmSignalRec2(i, self)
            self.schedule.add(a)
            x = 45
            y = 45
            a.location = (x, y)
            a.direction = -2.3561944901923448
            self.grid.add_object_to_grid((x, y), a)
            self.agents.append(a)

        # Define agent which is going to receive the signal
        for i in range(self.num_agents):
            a = SwarmSignalRec1(i, self)
            self.schedule.add(a)
            x = 0
            y = 0
            a.location = (x, y)
            a.direction = -2.3561944901923448
            self.grid.add_object_to_grid((x, y), a)
            self.agents.append(a)

    def step(self):
        self.schedule.step()


class TestSignalRecSwarmSmallGrid(TestCase):
    # Testing the functionality of signal. Signal needs to
    # broadcast information and move along with the agent. The
    # broadcasted information is about any object. Signal object
    # attaches the object for information transfer

    def setUp(self):
        self.environment = SignalModelRec(
            1, 100, 100, 10, 123)

    def test_agent_signal_send(self):

        for i in range(20):
            self.environment.step()

        agent = self.environment.agents[0]

        # Checking if the signal sending agent has reached the hub
        self.assertEqual(agent.location, (9, 9))

    def test_agent_signal_movement(self):

        for i in range(20):
            self.environment.step()

        agent = self.environment.agents[0]
        agent_loc = agent.location
        _, grid_val = self.environment.grid.find_grid(agent_loc)
        signal = self.environment.grid.get_objects('Signal', grid_val)

        # Checking is the signal is moving along with the agent
        self.assertEqual(signal, agent.signals)

    def test_agent_signal_receive(self):

        for i in range(30):
            self.environment.step()

        # Checking if the signal receiving agent has reached the site
        self.assertEqual(self.environment.agents[1].location, (20, 20))


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

        print('move', self.agent.name, self.agent.accleration,
        self.agent.velocity, self.agent.location, self.agent.direction)
        # Full carried object moves along the agent
        for item in self.agent.attached_objects:
            item.location = self.agent.location

        # Partially carried object is moved by the team effor
        self.update_partial_attached_objects()

        return Status.SUCCESS
"""
