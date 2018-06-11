"""Defines all the composite behaviors for the agents.

    This file name is scbehaviors coz `s` stands for swarms and
    `c` stands for composite behaviors.
    These composite behaviors are designed so that the algorithms
    would find the collective behaviors with ease. Also it will help the
    human designers to effectively design new behaviors. It provides
    flexibility for the user to use the primitive behaviors along with
    feature rich behaviors.
"""

from py_trees import Behaviour, Blackboard
from py_trees.composites import Sequence
from py_trees.trees import BehaviourTree
from swarms.sbehaviors import (
    GoTo, Towards, Move, Away,
    IsCarryable, IsSingleCarry, SingleCarry
    )

# Start of mid-level behaviors. These behaviors are the
# combination of primitive behaviors. There are behaviors which can
# make use of mid-level behavior to form highlevel behaviors.

# Every direction chaning command needs to follow move. So we will combine
# them into a single behaviors with sequence and call it MoveTowards


class MoveTowards(Behaviour):
    """MoveTowards behavior for the agents.

    Inherits the Behaviors class from py_trees. This
    behavior combines the privitive behaviors GoTo, Towards and Move. This
    allows agents actually to move towards the object of interest.
    """

    def __init__(self, name):
        """Init method for the MoveTowards behavior."""
        super(MoveTowards, self).__init__(name)
        self.blackboard = Blackboard()
        self.blackboard.shared_content = dict()

    def setup(self, timeout, agent, item):
        """Have defined the setup method.

        This method defines the other objects required for the
        behavior. Agent is the actor in the environment,
        item is the name of the item we are trying to find in the
        environment and timeout defines the execution time for the
        behavior.
        """
        self.agent = agent
        self.item = item
        # Define goto primitive behavior
        goto = GoTo('MT_GOTO_1')
        goto.setup(0, self.agent, self.item)

        # Define towards behavior
        towards = Towards('MT_TOWARDS_2')
        towards.setup(0, self.agent)

        # Define move behavior
        move = Move('MT_MOVE_3')
        move.setup(0, self.agent)

        # Define a sequence to combine the primitive behavior
        mt_sequence = Sequence('MT_SEQUENCE')
        mt_sequence.add_children([goto, towards, move])

        self.behaviour_tree = BehaviourTree(mt_sequence)

    def initialise(self):
        """Everytime initialization. Not required for now."""
        pass

    def update(self):
        """
        Just call the tick method for the sequence.
        This will execute the primitive behaviors defined in the sequence
        """
        self.behaviour_tree.tick()
        return self.behaviour_tree.root.status


class MoveAway(Behaviour):
    """MoveAway behavior for the agents.

    Inherits the Behaviors class from py_trees. This
    behavior combines the privitive behaviors GoTo, Away and Move. This
    allows agents actually to move away the object of interest.
    """

    def __init__(self, name):
        """Init method for the MoveAway behavior."""
        super(MoveAway, self).__init__(name)
        self.blackboard = Blackboard()
        self.blackboard.shared_content = dict()

    def setup(self, timeout, agent, item):
        """Have defined the setup method.

        This method defines the other objects required for the
        behavior. Agent is the actor in the environment,
        item is the name of the item we are trying to find in the
        environment and timeout defines the execution time for the
        behavior.
        """
        self.agent = agent
        self.item = item
        # Define goto primitive behavior
        goto = GoTo('MA_GOTO_1')
        goto.setup(0, self.agent, self.item)

        # Define away behavior
        away = Away('MA_Away_2')
        away.setup(0, self.agent)

        # Define move behavior
        move = Move('MA_MOVE_3')
        move.setup(0, self.agent)

        # Define a sequence to combine the primitive behavior
        mt_sequence = Sequence('MA_SEQUENCE')
        mt_sequence.add_children([goto, away, move])

        self.behaviour_tree = BehaviourTree(mt_sequence)

    def initialise(self):
        """Everytime initialization. Not required for now."""
        pass

    def update(self):
        """
        Just call the tick method for the sequence.
        This will execute the primitive behaviors defined in the sequence
        """
        self.behaviour_tree.tick()
        return self.behaviour_tree.root.status


class CompositeSingleCarry(Behaviour):
    """CompositeSingleCarry behavior for the agents.

    Inherits the Behaviors class from py_trees. This
    behavior combines the privitive behaviors to succesfully carry any
    carryable object. It combines IsCarrable, IsSingleCarry and SingleCarry
    primitive behaviors.
    """

    def __init__(self, name):
        """Init method for the CompositeSingleCarry behavior."""
        super(CompositeSingleCarry, self).__init__(name)
        self.blackboard = Blackboard()
        self.blackboard.shared_content = dict()

    def setup(self, timeout, agent, item):
        """Have defined the setup method.

        This method defines the other objects required for the
        behavior. Agent is the actor in the environment,
        item is the name of the item we are trying to find in the
        environment and timeout defines the execution time for the
        behavior.
        """
        self.agent = agent
        self.item = item

        # First check if the item is carrable?
        carryable = IsCarryable('SC_IsCarryable_1')
        carryable.setup(0, self.agent, self.item)

        # Then check if the item can be carried by a single agent
        issinglecarry = IsSingleCarry('SC_IsSingleCarry_2')
        issinglecarry.setup(0, self.agent, self.item)

        # Finally, carry the object
        singlecarry = SingleCarry('SC_SingleCarry_3')
        singlecarry.setup(0, self.agent, self.item)

        # Define a sequence to combine the primitive behavior
        sc_sequence = Sequence('SC_SEQUENCE')
        sc_sequence.add_children([carryable, issinglecarry, singlecarry])

        self.behaviour_tree = BehaviourTree(sc_sequence)

    def initialise(self):
        """Everytime initialization. Not required for now."""
        pass

    def update(self):
        """
        Just call the tick method for the sequence.
        This will execute the primitive behaviors defined in the sequence
        """
        self.behaviour_tree.tick()
        return self.behaviour_tree.root.status
