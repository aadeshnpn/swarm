from swarms.lib.agent import Agent
from py_trees.behaviour import Behaviour
from py_trees.trees import BehaviourTree
from py_trees.composites import Sequence
from py_trees import common, blackboard
import py_trees
import numpy as np


class HasMoney(Behaviour):
    def __init__(self, name):
        super(HasMoney, self).__init__(name)

    def setup(self, timeout, agent):
        self.agent = agent

    def initialise(self):
        pass

    def update(self):
        if self.agent.wealth > 0:
            return common.Status.SUCCESS
        else:
            return common.Status.FAILURE


class NeighbourCondition(Behaviour):
    def __init__(self, name):
        super(NeighbourCondition, self).__init__(name)

    def setup(self, timeout, agent):
        self.agent = agent

    def initialise(self):
        pass

    def update(self):
        cellmates = self.agent.model.grid.get_objects_from_grid('SwarmAgent', self.agent.location)
        if len(cellmates) > 1:
            return common.Status.SUCCESS
        else:
            return common.Status.FAILURE


class ShareMoney(Behaviour):
    def __init__(self, name):
        super(ShareMoney, self).__init__(name)

    def setup(self, timeout, agent):
        self.agent = agent

    def initialise(self):
        pass

    def update(self):
        try:
            cellmates = self.agent.model.grid.get_objects_from_grid('SwarmAgent', self.agent.location)
            others = self.agent.model.random.choice(cellmates)
            others.wealth += 1
            self.agent.wealth -= 1
            return common.Status.SUCCESS
        except:
            return common.Status.FAILURE


class Move(Behaviour):
    def __init__(self, name):
        super(Move, self).__init__(name)

    def setup(self, timeout, agent):
        self.agent = agent

    def initialise(self):
        pass

    def update(self):
        try:
            x = int(self.agent.location[0] + np.cos(self.agent.direction) * self.agent.speed)
            y = int(self.agent.location[1] + np.sin(self.agent.direction) * self.agent.speed)
            new_location, direction = self.agent.model.grid.check_limits((x, y), self.agent.direction)
            self.agent.model.grid.move_object(self.agent.location, self.agent, new_location)
            self.agent.location = new_location
            self.agent.direction = direction
            return common.Status.SUCCESS
        except:
            return common.Status.FAILURE


class SwarmAgent(Agent):
    """ An minimalistic swarm agent """
    def __init__(self, name, model):
        super().__init__(name, model)
        self.wealth = 1
        self.location = ()

        self.direction = model.random.rand() * (2 * np.pi)
        self.speed = 2
        self.radius = 3
        self.shared_content = dict()

        self.blackboard = blackboard.Client(name=str(name))
        self.blackboard.register_key(
            key='neighbourobj', access=common.Access.WRITE)
        self.blackboard.neighbourobj = dict()

        self.shared_content['Sites'] = {model.site}

        root = Sequence("Sequence")
        low = Move('4')
        low.setup(0, self)
        higest = HasMoney('1')
        higest.setup(0, self)
        high = NeighbourCondition('2')
        high.setup(0, self)
        med = ShareMoney('3')
        med.setup(0, self)

        root.add_children([low, higest, high, med])
        self.behaviour_tree = BehaviourTree(root)
        ## Visulalize the Behavior Tree
        # print(py_trees.display.ascii_tree(self.behaviour_tree.root))
        # py_trees.logging.level = py_trees.logging.Level.DEBUG

        # """
        # This above part should be replaced by Grammatical Evolution.
        # Based on research, use XML file to generate BT. Parse the XML BT
        # To actually get BT python program gm

    def step(self):
        # """
        # Doing this is equivalent of using behavior tree with four classes
        # in this order, Move, HasMoney, NeighbourCondition, ShareMoney
        # self.move()

        self.behaviour_tree.tick()

    def advance(self):
        pass