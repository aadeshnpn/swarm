from swarms.lib.agent import Agent

from py_trees.trees import BehaviourTree
from py_trees.composites import Sequence
from py_trees import common, blackboard
import py_trees
import numpy as np

from swarms.behaviors.scbehaviors import (
    MoveTowards
    )


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

        self.blackboard = blackboard.Client(name=str(name))
        self.blackboard.register_key(
            key='neighbourobj', access=common.Access.WRITE)
        self.blackboard.neighbourobj = dict()

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
        # print(py_trees.display.ascii_tree(self.behaviour_tree.root))
        # py_trees.logging.level = py_trees.logging.Level.DEBUG

    def step(self):
        self.behaviour_tree.tick()