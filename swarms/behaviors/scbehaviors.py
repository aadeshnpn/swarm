"""Defines all the composite behaviors for the agents.

This file name is scbehaviors coz `s` stands for swarms and
`c` stands for composite behaviors.
These composite behaviors are designed so that the algorithms
would find the collective behaviors with ease. Also it will help the
human designers to effectively design new behaviors. It provides
flexibility for the user to use the primitive behaviors along with
feature rich behaviors.
"""
import copy

from py_trees import decorators
from py_trees.trees import BehaviourTree
from py_trees.behaviour import Behaviour
from py_trees.composites import Sequence, Selector, Parallel
from py_trees import common, blackboard
from py_trees.decorators import FailureIsSuccess, Inverter

from swarms.behaviors.sbehaviors import (
    DropPheromone, GoTo, IsAttractivePheromone, IsDropable, IsMoveable, IsRepulsivePheromone, IsVisitedBefore, PheromoneExists, SensePheromone, Towards, Move, Away,
    IsCarryable, IsSingleCarry, SingleCarry,
    IsMultipleCarry, IsInPartialAttached, IsEnoughStrengthToCarry,
    InitiateMultipleCarry, IsCarrying, Drop, RandomWalk, DropPartial,
    SignalDoesNotExists, SendSignal, NeighbourObjects, ReceiveSignal,
    CueDoesNotExists, DropCue, PickCue, AvoidSObjects, NeighbourObjectsDist,
    IsSignalActive, IsAgentDead, CanMove
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
        # This is the post condition
        selector = Selector('MT_Selector')
        postcond_is_already_at_there = NeighbourObjects('MT_AlreadyThere_PC')
        postcond_is_already_at_there.setup(0, self.agent, self.item)

        goto = GoTo('MT_GOTO_1')
        goto.setup(0, self.agent, self.item)

        # Define towards behavior
        towards = Towards('MT_TOWARDS_2')
        towards.setup(0, self.agent)

        # This is the constraint/pre-condition
        const_is_no_blocked_obs = NeighbourObjectsDist('MT_Blocked_Obs_CNT')
        const_is_no_blocked_obs.setup(0, self.agent, 'Obstacles')

        const_is_no_blocked_trp = NeighbourObjectsDist('MT_Blocked_Trap_CNT')
        const_is_no_blocked_trp.setup(0, self.agent, 'Traps')

        # sequence_blocked = Sequence('MT_Blocked')
        sequence_blocked_obs = Sequence('MT_Selector_Blocked_Obs')
        avoid_obs = AvoidSObjects('MT_Avoid_Obstacles')
        avoid_obs.setup(0, self.agent)

        sequence_blocked_trp = Sequence('MT_Selector_Blocked_Trap')
        avoid_trp = AvoidSObjects('MT_Avoid_Traps')
        avoid_trp.setup(0, self.agent, item='Traps')

        # Define move behavior
        canmove = CanMove('MT_CANMOVE')
        canmove.setup(0, self.agent, None)

        move = Move('MT_MOVE_ACT')
        move.setup(0, self.agent)

        # selector_blocked.
        sequence_blocked_obs.add_children(
            [
                const_is_no_blocked_obs, avoid_obs, copy.copy(canmove), copy.copy(move)])
        sequence_blocked_trp.add_children(
            [
                const_is_no_blocked_trp, avoid_trp, copy.copy(canmove), copy.copy(move)])

        # Define a sequence to combine the primitive behavior
        mt_sequence = Sequence('MT_SEQUENCE')
        const_is_no_blocked_obs_inv = Inverter(copy.copy(const_is_no_blocked_obs))
        const_is_no_blocked_trp_inv = Inverter(copy.copy(const_is_no_blocked_trp))
        mt_sequence.add_children([goto, towards, const_is_no_blocked_obs_inv, const_is_no_blocked_trp_inv, canmove, move])

        selector.add_children([postcond_is_already_at_there, mt_sequence, sequence_blocked_obs, sequence_blocked_trp])

        self.behaviour_tree = BehaviourTree(selector)

    def initialise(self):
        """Everytime initialization. Not required for now."""
        pass

    def update(self):
        """Just call the tick method for the sequence.

        This will execute the primitive behaviors defined in the sequence
        """
        self.behaviour_tree.tick()
        return self.behaviour_tree.root.status


class MoveTowardsNormal(Behaviour):
    """MoveTowardsNormal behavior for the agents.

    Inherits the Behaviors class from py_trees. This
    behavior combines the privitive behaviors GoTo, Towards and Move. This
    allows agents actually to move towards the object of interest.
    """

    def __init__(self, name):
        """Init method for the MoveTowards behavior."""
        super(MoveTowardsNormal, self).__init__(name)

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
        # This is the post condition
        selector = Selector('MT_Selector')
        postcond_is_already_at_there = NeighbourObjects('MT_AlreadyThere_PC')
        postcond_is_already_at_there.setup(0, self.agent, self.item)

        goto = GoTo('MT_GOTO_1')
        goto.setup(0, self.agent, self.item)

        # Define towards behavior
        towards = Towards('MT_TOWARDS_2')
        towards.setup(0, self.agent)

        # Define move behavior
        canmove = CanMove('MT_CANMOVE')
        canmove.setup(0, self.agent, None)

        move = Move('MT_MOVE_ACT')
        move.setup(0, self.agent)

        # Define a sequence to combine the primitive behavior
        mt_sequence = Sequence('MT_SEQUENCE')

        mt_sequence.add_children([goto, towards, canmove, move])

        selector.add_children([postcond_is_already_at_there, mt_sequence])

        self.behaviour_tree = BehaviourTree(selector)

    def initialise(self):
        """Everytime initialization. Not required for now."""
        pass

    def update(self):
        """Just call the tick method for the sequence.

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

        # This is the post condition
        # Define goto primitive behavior
        # This is the post condition
        selector = Selector('MA_Selector')
        postcond_is_already_at_there = NeighbourObjects('MA_AlreadyThere_PC')
        postcond_is_already_at_there.setup(0, self.agent, self.item)

        goto = GoTo('MA_GOTO_1')
        goto.setup(0, self.agent, self.item)

        # Define towards behavior
        away = Away('MA_AWAY_2')
        away.setup(0, self.agent)

        # This is the constraint/pre-condition
        const_is_no_blocked_obs = NeighbourObjectsDist('MA_Blocked_Obs_CNT')
        const_is_no_blocked_obs.setup(0, self.agent, 'Obstacles')

        const_is_no_blocked_trp = NeighbourObjectsDist('MA_Blocked_Trap_CNT')
        const_is_no_blocked_trp.setup(0, self.agent, 'Traps')

        # sequence_blocked = Sequence('MA_Blocked')
        sequence_blocked_obs = Sequence('MA_Selector_Blocked_Obs')
        avoid_obs = AvoidSObjects('MA_Avoid_Obstacles')
        avoid_obs.setup(0, self.agent)

        sequence_blocked_trp = Sequence('MA_Selector_Blocked_Trap')
        avoid_trp = AvoidSObjects('MA_Avoid_Traps')
        avoid_trp.setup(0, self.agent, item='Traps')

        # Define move behavior
        canmove = CanMove('MA_CANMOVE')
        canmove.setup(0, self.agent, None)

        move = Move('MA_MOVE_ACT')
        move.setup(0, self.agent)

        # selector_blocked.
        sequence_blocked_obs.add_children([const_is_no_blocked_obs, avoid_obs, copy.copy(canmove), copy.copy(move)])
        sequence_blocked_trp.add_children([const_is_no_blocked_trp, avoid_trp, copy.copy(canmove), copy.copy(move)])

        # Define a sequence to combine the primitive behavior
        mt_sequence = Sequence('MA_SEQUENCE')
        const_is_no_blocked_obs_inv = Inverter(copy.copy(const_is_no_blocked_obs))
        const_is_no_blocked_trp_inv = Inverter(copy.copy(const_is_no_blocked_trp))
        mt_sequence.add_children([goto, away, const_is_no_blocked_obs_inv, const_is_no_blocked_trp_inv, canmove, move])

        selector.add_children([postcond_is_already_at_there, mt_sequence, sequence_blocked_obs, sequence_blocked_trp])

        self.behaviour_tree = BehaviourTree(selector)

    def initialise(self):
        """Everytime initialization. Not required for now."""
        pass

    def update(self):
        """Just call the tick method for the sequence.

        This will execute the primitive behaviors defined in the sequence
        """
        self.behaviour_tree.tick()
        return self.behaviour_tree.root.status


class MoveAwayNormal(Behaviour):
    """MoveAwayNormal behavior for the agents.

    Inherits the Behaviors class from py_trees. This
    behavior combines the privitive behaviors GoTo, Away and Move. This
    allows agents actually to move away the object of interest.
    """

    def __init__(self, name):
        """Init method for the MoveAway behavior."""
        super(MoveAwayNormal, self).__init__(name)

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

        # This is the post condition
        # Define goto primitive behavior
        # This is the post condition
        selector = Selector('MA_Selector')
        postcond_is_already_at_there = NeighbourObjects('MA_AlreadyThere_PC')
        postcond_is_already_at_there.setup(0, self.agent, self.item)

        goto = GoTo('MA_GOTO_1')
        goto.setup(0, self.agent, self.item)

        # Define towards behavior
        away = Away('MA_AWAY_2')
        away.setup(0, self.agent)

        # Define move behavior
        canmove = CanMove('MA_CANMOVE')
        canmove.setup(0, self.agent, None)

        move = Move('MA_MOVE_ACT')
        move.setup(0, self.agent)

        # Define a sequence to combine the primitive behavior
        mt_sequence = Sequence('MA_SEQUENCE')
        mt_sequence.add_children([goto, away, canmove, move])

        selector.add_children([postcond_is_already_at_there, mt_sequence])

        self.behaviour_tree = BehaviourTree(selector)

    def initialise(self):
        """Everytime initialization. Not required for now."""
        pass

    def update(self):
        """Just call the tick method for the sequence.

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

        # PostCondition
        selector = Selector('SC_Selector')
        is_already_carrying = IsCarrying('SC_IsCarrying_PC')
        is_already_carrying.setup(0, self.agent, self.item)

        # Check if there is item to be carried
        neigbourobjs = NeighbourObjects('SC_NeighbourObjects')
        neigbourobjs.setup(0, self.agent, self.item)

        # First check if the item is carrable?
        carryable = IsCarryable('SC_IsCarryable_1_CNT')
        carryable.setup(0, self.agent, self.item)

        # Then check if the item can be carried by a single agent
        issinglecarry = IsSingleCarry('SC_IsSingleCarry_2_CNT')
        issinglecarry.setup(0, self.agent, self.item)

        # Finally, carry the object
        singlecarry = SingleCarry('SC_SingleCarry_3')
        singlecarry.setup(0, self.agent, self.item)

        # Define a sequence to combine the primitive behavior
        sc_sequence = Sequence('SC_SEQUENCE')
        sc_sequence.add_children([neigbourobjs, carryable, issinglecarry, singlecarry])
        selector.add_children([is_already_carrying, sc_sequence])

        self.behaviour_tree = BehaviourTree(selector)

    def initialise(self):
        """Everytime initialization. Not required for now."""
        pass

    def update(self):
        """Just call the tick method for the sequence.

        This will execute the primitive behaviors defined in the sequence
        """
        self.behaviour_tree.tick()
        return self.behaviour_tree.root.status


# class CompositeMultipleCarry(Behaviour):
#     """CompositeMultipleCarry behavior for the agents.

#     Inherits the Behaviors class from py_trees. This
#     behavior combines the privitive behaviors to succesfully carry any heavy
#     carryable object. It combines IsCarrable, IsMultipleCarry
#     and InitiateMultipleCarry primitive behaviors.
#     """

#     def __init__(self, name):
#         """Init method for the CompositeSingleCarry behavior."""
#         super(CompositeMultipleCarry, self).__init__(name)

#     def setup(self, timeout, agent, item):
#         """Have defined the setup method.

#         This method defines the other objects required for the
#         behavior. Agent is the actor in the environment,
#         item is the name of the item we are trying to find in the
#         environment and timeout defines the execution time for the
#         behavior.
#         """
#         self.agent = agent
#         self.item = item

#         # Root node from the multiple carry behavior tree
#         root = Sequence("MC_Sequence")

#         # Conditional behavior to check if the sensed object is carrable or not
#         carryable = IsCarryable('MC_IsCarryable')
#         carryable.setup(0, self.agent, self.item)

#         # Conditional behavior to check if the object is too heavy
#         # for single carry
#         is_mc = IsMultipleCarry('MC_IsMultipleCarry')
#         is_mc.setup(0, self.agent, self.item)

#         # Check if the object is alread attached to the object
#         partial_attached = IsInPartialAttached('MC_IsPartialAttached')
#         partial_attached.setup(0, self.agent, self.item)

#         # Initiate multiple carry process
#         initiate_mc_b = InitiateMultipleCarry('MC_InitiateMultipleCarry')
#         initiate_mc_b.setup(0, self.agent, self.item)

#         # Selector to select between intiate
#         # multiple carry and checking strength
#         initial_mc_sel = Selector("MC_Selector")
#         initial_mc_sel.add_children([partial_attached, initiate_mc_b])

#         strength = IsEnoughStrengthToCarry('MC_EnoughStrength')
#         strength.setup(0, self.agent, self.item)

#         strength_seq = Sequence("MC_StrenghtSeq")

#         strength_seq.add_children([strength])

#         # Main sequence branch where all the multiple carry logic takes place
#         sequence_branch = Sequence("MC_Sequence_branch")
#         sequence_branch.add_children([is_mc, initial_mc_sel, strength_seq])

#         # Main logic behind this composite multiple carry BT
#         """
#         First check if the object is carryable or not. If the object is
#         carryable then execute the sequence branch. In the sequence branch,
#         check is the object needs multiple agents to carry. If yes, execute
#         the initiate multiple carry sequence branch only if it has not been
#         attached before. Finally, check if there are enought agents/strenght
#         to lift the object up.
#         """
#         root.add_children([carryable, sequence_branch])
#         self.behaviour_tree = BehaviourTree(root)

#     def initialise(self):
#         """Everytime initialization. Not required for now."""
#         pass

#     def update(self):
#         """Just call the tick method for the sequence.

#         This will execute the primitive behaviors defined in the sequence
#         """
#         self.behaviour_tree.tick()
#         return self.behaviour_tree.root.status


class CompositeDrop(Behaviour):
    """CompositeDrop behavior for the agents.

    Inherits the Behaviors class from py_trees. This
    behavior combines the privitive behaviors to succesfully drop any
    carrying object fon to a dropable surface. It combines IsDropable,
    IsCarrying and Drop primitive behaviors.
    """

    def __init__(self, name):
        """Init method for the CompositeDrop behavior."""
        super(CompositeDrop, self).__init__(name)

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

        # PostCondition
        selector = Selector('SC_Selector')
        is_carrying = IsCarrying('SC_IsCarrying_PC')
        is_carrying.setup(0, self.agent, self.item)
        is_dropped = Inverter(is_carrying)

        dropseq = Sequence('CD_Sequence')
        # Check if there is item to be carried
        neigbourobjs = NeighbourObjects('CD_NeighbourObjects')
        neigbourobjs.setup(0, self.agent, None)
        neigbourobjs = FailureIsSuccess(neigbourobjs)

        iscarrying = IsCarrying('CD_IsCarrying_CNT')
        iscarrying.setup(0, self.agent, self.item)

        isdropable = IsDropable('CD_Dropable_CNT')
        isdropable.setup(0, self.agent, None)

        drop = Drop('CD_Drop')
        drop.setup(0, self.agent, self.item)

        dropseq.add_children([neigbourobjs, iscarrying, isdropable, drop])
        selector.add_children([is_dropped, dropseq])

        self.behaviour_tree = BehaviourTree(dropseq)

    def initialise(self):
        """Everytime initialization. Not required for now."""
        pass

    def update(self):
        """Just call the tick method for the sequence.

        This will execute the primitive behaviors defined in the sequence
        """
        self.behaviour_tree.tick()
        return self.behaviour_tree.root.status


# class CompositeDropPartial(Behaviour):
#     """CompositeDropPartial behavior for the agents.

#     Inherits the Behaviors class from py_trees. This
#     behavior combines the privitive behaviors to succesfully drop any
#     objects carried with cooperation to a dropable surface. It
#     combines IsDropable, IsCarrying and DropPartial primitive behaviors.
#     """

#     def __init__(self, name):
#         """Init method for the CompositeDrop behavior."""
#         super(CompositeDropPartial, self).__init__(name)

#     def setup(self, timeout, agent, item):
#         """Have defined the setup method.

#         This method defines the other objects required for the
#         behavior. Agent is the actor in the environment,
#         item is the name of the item we are trying to find in the
#         environment and timeout defines the execution time for the
#         behavior.
#         """
#         self.agent = agent
#         self.item = item

#         dropseq = Sequence('CDP_Sequence')

#         iscarrying = IsInPartialAttached('CDP_IsInPartial')
#         iscarrying.setup(0, self.agent, self.item)

#         drop = DropPartial('CDP_DropPartial')
#         drop.setup(0, self.agent, self.item)

#         dropseq.add_children([iscarrying, drop])

#         self.behaviour_tree = BehaviourTree(dropseq)

#     def initialise(self):
#         """Everytime initialization. Not required for now."""
#         pass

#     def update(self):
#         """Just call the tick method for the sequence.

#         This will execute the primitive behaviors defined in the sequence
#         """
#         self.behaviour_tree.tick()
#         return self.behaviour_tree.root.status


class Explore(Behaviour):
    """Explore behavior for the agents.

    Inherits the Behaviors class from py_trees. This
    behavior combines the privitive behaviors to succesfully explore the
    environment. It combines Randomwalk and Move behaviors.
    """

    def __init__(self, name):
        """Init method for the Explore behavior."""
        super(Explore, self).__init__(name)

    def setup(self, timeout, agent, item=None):
        """Have defined the setup method.

        This method defines the other objects required for the
        behavior. Agent is the actor in the environment,
        item is the name of the item we are trying to find in the
        environment and timeout defines the execution time for the
        behavior.
        """
        self.agent = agent
        self.item = item

        # Define the root for the BT
        root = Selector('Explore_root')
        ex_sequence = Sequence("Ex_Sequence")

        low = RandomWalk('Ex_RandomWalk')
        low.setup(0, self.agent)

        # This is the constraint/pre-condition
        const_is_no_blocked_obs = NeighbourObjectsDist('EX_Blocked_Obs_CNT')
        const_is_no_blocked_obs.setup(0, self.agent, 'Obstacles')

        const_is_no_blocked_trp = NeighbourObjectsDist('EX_Blocked_Trap_CNT')
        const_is_no_blocked_trp.setup(0, self.agent, 'Traps')

        # sequence_blocked = Sequence('EX_Blocked')
        sequence_blocked_obs = Sequence('EX_Selector_Blocked_Obs')
        avoid_obs = AvoidSObjects('EX_Avoid_Obstacles')
        avoid_obs.setup(0, self.agent)

        sequence_blocked_trp = Sequence('EX_Selector_Blocked_Trap')
        avoid_trp = AvoidSObjects('EX_Avoid_Traps')
        avoid_trp.setup(0, self.agent, item='Traps')

        canmove = CanMove('Ex_CANMOVE')
        canmove.setup(0, self.agent, None)

        high = Move('Ex_Move')
        high.setup(0, self.agent)

        # selector_blocked.
        const_is_no_blocked_obs_inv = Inverter(copy.copy(const_is_no_blocked_obs))
        const_is_no_blocked_trp_inv = Inverter(copy.copy(const_is_no_blocked_trp))
        sequence_blocked_obs.add_children([const_is_no_blocked_obs, avoid_obs, copy.copy(canmove), copy.copy(high)])
        sequence_blocked_trp.add_children([const_is_no_blocked_trp, avoid_trp, copy.copy(canmove), copy.copy(high)])

        ex_sequence.add_children([low, const_is_no_blocked_obs_inv, const_is_no_blocked_trp_inv, canmove, high])

        root.add_children([ex_sequence, sequence_blocked_obs, sequence_blocked_trp])

        self.behaviour_tree = BehaviourTree(root)

    def initialise(self):
        """Everytime initialization. Not required for now."""
        pass

    def update(self):
        """Just call the tick method for the sequence.

        This will execute the primitive behaviors defined in the sequence
        """
        self.behaviour_tree.tick()
        return self.behaviour_tree.root.status


class ExploreNormal(Behaviour):
    """ExploreNormal behavior for the agents.

    Inherits the Behaviors class from py_trees. This
    behavior combines the privitive behaviors to succesfully explore the
    environment. It combines Randomwalk and Move behaviors.
    """

    def __init__(self, name):
        """Init method for the Explore behavior."""
        super(ExploreNormal, self).__init__(name)

    def setup(self, timeout, agent, item=None):
        """Have defined the setup method.

        This method defines the other objects required for the
        behavior. Agent is the actor in the environment,
        item is the name of the item we are trying to find in the
        environment and timeout defines the execution time for the
        behavior.
        """
        self.agent = agent
        self.item = item

        # Define the root for the BT
        root = Sequence("Ex_Sequence")

        low = RandomWalk('Ex_RandomWalk')
        low.setup(0, self.agent)

        canmove = CanMove('Ex_CANMOVE')
        canmove.setup(0, self.agent, None)

        high = Move('Ex_Move')
        high.setup(0, self.agent)

        root.add_children([low, canmove, high])
        self.behaviour_tree = BehaviourTree(root)

    def initialise(self):
        """Everytime initialization. Not required for now."""
        pass

    def update(self):
        """Just call the tick method for the sequence.

        This will execute the primitive behaviors defined in the sequence
        """
        self.behaviour_tree.tick()
        return self.behaviour_tree.root.status


# Communication composite behaviors
class CompositeSendSignal(Behaviour):
    """Send signal behavior for the agents.

    Inherits the Behaviors class from py_trees. This
    behavior combines the privitive behaviors to succesfully send signals
    in the environment. It combines SignalDoesNotExists and SendSignal
    behaviors.
    """

    def __init__(self, name):
        """Init method for the SendSignal behavior."""
        super(CompositeSendSignal, self).__init__(name)

    def setup(self, timeout, agent, item=None):
        """Have defined the setup method.

        This method defines the other objects required for the
        behavior. Agent is the actor in the environment,
        item is the name of the item we are trying to find in the
        environment and timeout defines the execution time for the
        behavior.
        """
        self.agent = agent
        self.item = item

        # Define the root for the BT
        root = Selector('CSS_root')
        is_signal_active = IsSignalActive('CSS_IsSignalActive')
        is_signal_active.setup(0, self.agent, None)

        sequencesig = Sequence('CSS_Sequence')
        neigbourobjs = IsVisitedBefore('CSS_VisitedBefore')
        neigbourobjs.setup(0, self.agent, self.item)

        s2 = SendSignal('CSS_SendSignal')
        s2.setup(0, self.agent, self.item)

        sequencesig.add_children([neigbourobjs, s2])

        root.add_children([is_signal_active, sequencesig])
        self.behaviour_tree = BehaviourTree(root)

    def initialise(self):
        """Everytime initialization. Not required for now."""
        pass

    def update(self):
        """Just call the tick method for the sequence.

        This will execute the primitive behaviors defined in the sequence
        """
        self.behaviour_tree.tick()
        return self.behaviour_tree.root.status


class CompositeReceiveSignal(Behaviour):
    """Receive signal behavior for the agents.

    Inherits the Behaviors class from py_trees. This
    behavior combines the privitive behaviors to succesfully receive signals
    in the environment. It combines Neighbour and ReceiveSignal behaviors.
    """

    def __init__(self, name):
        """Init method for the SendSignal behavior."""
        super(CompositeReceiveSignal, self).__init__(name)

    def setup(self, timeout, agent, item=None):
        """Have defined the setup method.

        This method defines the other objects required for the
        behavior. Agent is the actor in the environment,
        item is the name of the item we are trying to find in the
        environment and timeout defines the execution time for the
        behavior.
        """
        self.agent = agent
        self.item = item

        # Define the root for the BT
        root = Selector('CRS_root')

        seqsignal = Sequence('CRS_Sequence')

        neigbourobjs = IsVisitedBefore('CRS_VisitedBefore')
        neigbourobjs.setup(0, self.agent, 'Signal')

        s1 = NeighbourObjects('CRS_NeighbourObjects')
        s1.setup(0, self.agent, 'Signal')

        s2 = ReceiveSignal('CRS_ReceiveSignal')
        s2.setup(0, self.agent, 'Signal')

        seqsignal.add_children([s1, s2])
        root.add_children([neigbourobjs, seqsignal])

        self.behaviour_tree = BehaviourTree(root)

    def initialise(self):
        """Everytime initialization. Not required for now."""
        pass

    def update(self):
        """Just call the tick method for the sequence.

        This will execute the primitive behaviors defined in the sequence
        """
        self.behaviour_tree.tick()
        return self.behaviour_tree.root.status


# class CompositeDropCue(Behaviour):
#     """Drop cue behavior for the agents.

#     Inherits the Behaviors class from py_trees. This
#     behavior combines the privitive behaviors to succesfully drop cue
#     in the environment. It combines CueDoesNotExists and DropCue behaviors.
#     """

#     def __init__(self, name):
#         """Init method for the SendSignal behavior."""
#         super(CompositeDropCue, self).__init__(name)

#     def setup(self, timeout, agent, item=None):
#         """Have defined the setup method.

#         This method defines the other objects required for the
#         behavior. Agent is the actor in the environment,
#         item is the name of the item we are trying to find in the
#         environment and timeout defines the execution time for the
#         behavior.
#         """
#         self.agent = agent
#         self.item = item

#         # Define the root for the BT
#         root = Sequence('CDC_Sequence')

#         c1 = CueDoesNotExists('CDC_CueDoesNotExists')
#         c1.setup(0, self.agent, self.item)

#         c2 = DropCue('CDC_DropCue')
#         c2.setup(0, self.agent, self.item)

#         root.add_children([c1, c2])

#         self.behaviour_tree = BehaviourTree(root)

#     def initialise(self):
#         """Everytime initialization. Not required for now."""
#         pass

#     def update(self):
#         """Just call the tick method for the sequence.

#         This will execute the primitive behaviors defined in the sequence
#         """
#         self.behaviour_tree.tick()
#         return self.behaviour_tree.root.status


# class CompositePickCue(Behaviour):
#     """Pick cue behavior for the agents.

#     Inherits the Behaviors class from py_trees. This
#     behavior combines the privitive behaviors to succesfully pick cue
#     from the environment. It combines NeighbourObjects and PickCue behaviors.
#     """

#     def __init__(self, name):
#         """Init method for the SendSignal behavior."""
#         super(CompositePickCue, self).__init__(name)

#     def setup(self, timeout, agent, item=None):
#         """Have defined the setup method.

#         This method defines the other objects required for the
#         behavior. Agent is the actor in the environment,
#         item is the name of the item we are trying to find in the
#         environment and timeout defines the execution time for the
#         behavior.
#         """
#         self.agent = agent
#         self.item = item

#         # Define the root for the BT
#         root = Sequence('CPC_Sequence')

#         c1 = NeighbourObjects('CPS_SearchCue')
#         c1.setup(0, self.agent, 'Cue')

#         c2 = PickCue('CPS_PickCue')
#         c2.setup(0, self.agent, 'Cue')

#         root.add_children([c1, c2])

#         self.behaviour_tree = BehaviourTree(root)

#     def initialise(self):
#         """Everytime initialization. Not required for now."""
#         pass

#     def update(self):
#         """Just call the tick method for the sequence.

#         This will execute the primitive behaviors defined in the sequence
#         """
#         self.behaviour_tree.tick()
#         return self.behaviour_tree.root.status


class CompositeDropPheromone(Behaviour):
    """Drop pheromone behavior for the agents.

    Inherits the Behaviors class from py_trees. This
    behavior combines the privitive behaviors to succesfully drop pheromone
    int the environment. It combines NeighbourObjects and DroPheromones behaviors.
    """

    def __init__(self, name):
        """Init method for the SendSignal behavior."""
        super(CompositeDropPheromone, self).__init__(name)

    def setup(self, timeout, agent, item=None):
        """Have defined the setup method.

        This method defines the other objects required for the
        behavior. Agent is the actor in the environment,
        item is the name of the item we are trying to find in the
        environment and timeout defines the execution time for the
        behavior.
        """
        self.agent = agent
        self.item = item

        # Define the root for the BT
        root = Selector('CDP_Selector')

        seqagent_dead = Sequence('CDP_Sequence_Dead')
        seqagent_notdead = Sequence('CDP_Sequence_NotDead')

        exists = Sequence('CPD_PExists')
        neigh = NeighbourObjects('CDP_PheromoneNearBy')
        neigh.setup(0, self.agent, 'Pheromones')

        pheromone_exists = PheromoneExists('CDP_PheromoneExists')
        pheromone_exists.setup(0, self.agent, 'Pheromones')

        exists.add_children([neigh, pheromone_exists])

        c1 = IsDropable('CDP_Dropable1')
        c1.setup(0, self.agent, None)

        c11 = IsDropable('CDP_Dropable2')
        c11.setup(0, self.agent, None)

        c2 = IsAgentDead('CDP_IsAgentDead')
        c2.setup(0, self.agent, None)

        c3 =  Inverter(copy.copy(c2), 'CDP_AgentNotDead')

        c4 = DropPheromone('CDP_DropPheromone_Att', attractive=True)
        c4.setup(0, self.agent, 'Cue')
        c5 = DropPheromone('CDP_DropPheromone_Rep', attractive=False)
        c5.setup(0, self.agent, 'Cue')

        seqagent_dead.add_children([c1, c2, c5])
        seqagent_notdead.add_children([c11, c3, c4])

        root.add_children([exists, seqagent_notdead, seqagent_dead])

        self.behaviour_tree = BehaviourTree(root)

    def initialise(self):
        """Everytime initialization. Not required for now."""
        pass

    def update(self):
        """Just call the tick method for the sequence.

        This will execute the primitive behaviors defined in the sequence
        """
        self.behaviour_tree.tick()
        return self.behaviour_tree.root.status


class CompositeSensePheromone(Behaviour):
    """Sense pheromone behavior for the agents.

    Inherits the Behaviors class from py_trees. This
    behavior combines the privitive behaviors to succesfully sense pheromone
    in the environment. It combines NeighbourObjects and SensePheromones behaviors.
    """

    def __init__(self, name):
        """Init method for the CompositeSensePheromone behavior."""
        super(CompositeSensePheromone, self).__init__(name)

    def setup(self, timeout, agent, item=None):
        """Have defined the setup method.

        This method defines the other objects required for the
        behavior. Agent is the actor in the environment,
        item is the name of the item we are trying to find in the
        environment and timeout defines the execution time for the
        behavior.
        """
        self.agent = agent
        self.item = item

        # Define the root for the BT
        root = Selector('CSP_Selector')

        seqagent_pheromon = Sequence('CSP_Sequence')

        neigh = NeighbourObjects('CSP_PheromoneNearBy')
        neigh.setup(0, self.agent, 'Pheromones')

        is_attractive_phe = IsAttractivePheromone('CSP_IsAttractive')
        is_attractive_phe.setup(0, self.agent, 'Pheromones')

        sense_pheromonc = SensePheromone('CSP_SensePheromon')
        sense_pheromonc.setup(0, self.agent, 'Pheromones')

        canmove = CanMove('CSP_CanMove')
        canmove.setup(0, self.agent, None)

        move = Move('CSP_Move')
        move.setup(0, self.agent, None)

        seqagent_pheromon.add_children([neigh, is_attractive_phe, sense_pheromonc, copy.copy(canmove), copy.copy(move)])

        repuslive_pehromon = Sequence('CSP_Sequence_Replusive')

        neighr = NeighbourObjectsDist('CSP_PheromoneNearByR')
        neighr.setup(0, self.agent, 'Pheromones')

        is_repulsive_phe = IsRepulsivePheromone('CSP_IsRepulsive')
        is_repulsive_phe.setup(0, self.agent, 'Pheromones')

        avoid = AvoidSObjects('CSP_Avoid')
        avoid.setup(0, self.agent, 'Pheromones')

        repuslive_pehromon.add_children([neighr, is_repulsive_phe, avoid, canmove, move])
        root.add_children([seqagent_pheromon, repuslive_pehromon])

        self.behaviour_tree = BehaviourTree(root)

    def initialise(self):
        """Everytime initialization. Not required for now."""
        pass

    def update(self):
        """Just call the tick method for the sequence.

        This will execute the primitive behaviors defined in the sequence
        """
        self.behaviour_tree.tick()
        return self.behaviour_tree.root.status
