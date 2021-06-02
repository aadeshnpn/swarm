"""Defines all the composite behaviors for the agents.

This file name is scbehaviors coz `s` stands for swarms and
`c` stands for composite behaviors.
These composite behaviors are designed so that the algorithms
would find the collective behaviors with ease. Also it will help the
human designers to effectively design new behaviors. It provides
flexibility for the user to use the primitive behaviors along with
feature rich behaviors.
"""

from py_trees import decorators
from py_trees.trees import BehaviourTree
from py_trees.behaviour import Behaviour
from py_trees.composites import Sequence, Selector, Parallel
from py_trees import common, blackboard
from py_trees.decorators import FailureIsSuccess, Inverter

from swarms.behaviors.sbehaviors import (
    GoTo, IsDropable, IsMoveable, Towards, Move, Away,
    IsCarryable, IsSingleCarry, SingleCarry,
    IsMultipleCarry, IsInPartialAttached, IsEnoughStrengthToCarry,
    InitiateMultipleCarry, IsCarrying, Drop, RandomWalk, DropPartial,
    SignalDoesNotExists, SendSignal, NeighbourObjects, ReceiveSignal,
    CueDoesNotExists, DropCue, PickCue, AvoidSObjects, NeighbourObjectsDist
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
        const_is_no_blocked_obs_inv = Inverter(const_is_no_blocked_obs)

        const_is_no_blocked_trp = NeighbourObjectsDist('MT_Blocked_Trap_CNT')
        const_is_no_blocked_trp.setup(0, self.agent, 'Traps')
        const_is_no_blocked_trp_inv = Inverter(const_is_no_blocked_trp)

        sequence_blocked = Sequence('MT_Blocked')
        selector_blocked_obs = Selector('MT_Selector_Blocked_Obs')
        avoid_obs = AvoidSObjects('MT_Avoid_Obstacles')
        avoid_obs.setup(0, self.agent)

        selector_blocked_trp = Selector('MT_Selector_Blocked_Trap')
        avoid_trp = AvoidSObjects('MT_Avoid_Traps')
        avoid_trp.setup(0, self.agent, item='Traps')

        # selector_blocked.
        selector_blocked_obs.add_children([const_is_no_blocked_obs_inv, avoid_obs])
        selector_blocked_trp.add_children([const_is_no_blocked_trp_inv, avoid_trp])
        sequence_blocked.add_children([selector_blocked_obs, selector_blocked_trp])

        # Define move behavior
        move = Move('MT_MOVE_ACT')
        move.setup(0, self.agent)

        # Define a sequence to combine the primitive behavior
        mt_sequence = Sequence('MT_SEQUENCE')
        mt_sequence.add_children([goto, towards, sequence_blocked, move])

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
        away.setup(0, self.agent, None)

        # This is the constraint/pre-condition
        const_is_no_blocked_obs = NeighbourObjectsDist('MA_Blocked_Obs_CNT')
        const_is_no_blocked_obs.setup(0, self.agent, 'Obstacles')
        const_is_no_blocked_obs_inv = Inverter(const_is_no_blocked_obs)

        const_is_no_blocked_trp = NeighbourObjectsDist('MA_Blocked_Trap_CNT')
        const_is_no_blocked_trp.setup(0, self.agent, 'Traps')
        const_is_no_blocked_trp_inv = Inverter(const_is_no_blocked_trp)

        sequence_blocked = Sequence('MA_Blocked')
        selector_blocked_obs = Selector('MA_Selector_Blocked_Obs')
        avoid_obs = AvoidSObjects('MA_Avoid_Obstacles')
        avoid_obs.setup(0, self.agent)

        selector_blocked_trp = Selector('MA_Selector_Blocked_Trap')
        avoid_trp = AvoidSObjects('MA_Avoid_Traps')
        avoid_trp.setup(0, self.agent, item='Traps')

        # selector_blocked.
        selector_blocked_obs.add_children([const_is_no_blocked_obs_inv, avoid_obs])
        selector_blocked_trp.add_children([const_is_no_blocked_trp_inv, avoid_trp])
        sequence_blocked.add_children([selector_blocked_obs, selector_blocked_trp])

        # Define move behavior
        move = Move('MA_MOVE_ACT')
        move.setup(0, self.agent)

        # Define a sequence to combine the primitive behavior
        mt_sequence = Sequence('MA_SEQUENCE')
        mt_sequence.add_children([goto, away, sequence_blocked, move])

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
        root = Sequence("Ex_Sequence")

        low = RandomWalk('Ex_RandomWalk')
        low.setup(0, self.agent)

        high = Move('Ex_Move')
        high.setup(0, self.agent)

        root.add_children([low, high])

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
        root = Sequence('CSS_Sequence')

        s1 = SignalDoesNotExists('CSS_SignalDoesNotExists')
        s1.setup(0, self.agent, self.item)

        s2 = SendSignal('CSS_SendSignal')
        s2.setup(0, self.agent, self.item)

        root.add_children([s1, s2])

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
        root = Sequence('CRS_Sequence')

        s1 = NeighbourObjects('CRS_NeighbourObjects')
        s1.setup(0, self.agent, 'Signal')

        s2 = ReceiveSignal('CRS_ReceiveSignal')
        s2.setup(0, self.agent, 'Signal')

        root.add_children([s1, s2])

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


class CompositeDropCue(Behaviour):
    """Drop cue behavior for the agents.

    Inherits the Behaviors class from py_trees. This
    behavior combines the privitive behaviors to succesfully drop cue
    in the environment. It combines CueDoesNotExists and DropCue behaviors.
    """

    def __init__(self, name):
        """Init method for the SendSignal behavior."""
        super(CompositeDropCue, self).__init__(name)

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
        root = Sequence('CDC_Sequence')

        c1 = CueDoesNotExists('CDC_CueDoesNotExists')
        c1.setup(0, self.agent, self.item)

        c2 = DropCue('CDC_DropCue')
        c2.setup(0, self.agent, self.item)

        root.add_children([c1, c2])

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


class CompositePickCue(Behaviour):
    """Pick cue behavior for the agents.

    Inherits the Behaviors class from py_trees. This
    behavior combines the privitive behaviors to succesfully pick cue
    from the environment. It combines NeighbourObjects and PickCue behaviors.
    """

    def __init__(self, name):
        """Init method for the SendSignal behavior."""
        super(CompositePickCue, self).__init__(name)

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
        root = Sequence('CPC_Sequence')

        c1 = NeighbourObjects('CPS_SearchCue')
        c1.setup(0, self.agent, 'Cue')

        c2 = PickCue('CPS_PickCue')
        c2.setup(0, self.agent, 'Cue')

        root.add_children([c1, c2])

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


class AvoidTrapObstaclesBehaviour(Behaviour):
    """Avoid both obstacles and trap for the agents.

    Inherits the Behaviors class from py_trees. This
    behavior combines the privitive behaviors to succesfully avoid traps
    and obstacles in the environment.
    """

    def __init__(self, name):
        """Init method for the AvoidTrapObstacles behavior."""
        super(AvoidTrapObstaclesBehaviour, self).__init__(name)

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
        root = Sequence('ATO_Sequence')

        # m1 = NeighbourObjects('ATO_SearchTrap')
        m1 = NeighbourObjectsDist('ATO_Search')
        m1.setup(0, self.agent, item=None)
        m2 = AvoidSObjects('ATO_AvoidTrap')
        m2.setup(0, self.agent, 'Traps')
        # m3 = NeighbourObjects('ATO_SearchObstacles')
        # m3.setup(0, self.agent, item='Obstacles')
        m4 = AvoidSObjects('ATO_AvoidObstacle')
        m4.setup(0, self.agent, 'Obstacles')

        root.add_children([m1, m2, m4])
        # root.add_children([m3, m4])

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


class NewMoveTowards(Behaviour):
    """MoveTowards behavior for the agents.

    Inherits the Behaviors class from py_trees. This
    behavior combines the privitive behaviors GoTo, Towards and Move. This
    allows agents actually to move towards the object of interest.
    """

    def __init__(self, name):
        """Init method for the MoveTowards behavior."""
        super(NewMoveTowards, self).__init__(name)

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

        # Avoid Traps and Obstacles
        avoidto = AvoidTrapObstaclesBehaviour('MT_AVOID_3')
        avoidto.setup(0, self.agent)
        avoidto = FailureIsSuccess(avoidto)

        # Define move behavior
        move = Move('MT_MOVE_4')
        move.setup(0, self.agent)

        # Define a sequence to combine the primitive behavior
        mt_sequence = Sequence('MT_SEQUENCE')
        mt_sequence.add_children([goto, towards, avoidto, move])

        self.behaviour_tree = BehaviourTree(mt_sequence)

    def initialise(self):
        """Everytime initialization. Not required for now."""
        pass

    def update(self):
        """Just call the tick method for the sequence.

        This will execute the primitive behaviors defined in the sequence
        """
        self.behaviour_tree.tick()
        return self.behaviour_tree.root.status


class NewExplore(Behaviour):
    """Explore behavior for the agents.

    Inherits the Behaviors class from py_trees. This
    behavior combines the privitive behaviors to succesfully explore the
    environment. It combines Randomwalk and Move behaviors.
    """

    def __init__(self, name):
        """Init method for the Explore behavior."""
        super(NewExplore, self).__init__(name)

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

        # Avoid Traps and Obstacles
        avoidto = AvoidTrapObstaclesBehaviour('EX_AVOID')
        avoidto.setup(0, self.agent)
        avoidto = FailureIsSuccess(avoidto)

        high = Move('Ex_Move')
        high.setup(0, self.agent)

        root.add_children([low, avoidto, high])

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


class NewMoveAway(Behaviour):
    """NewMoveAway behavior for the agents.

    Inherits the Behaviors class from py_trees. This
    behavior combines the privitive behaviors GoTo, Away and Move. This
    allows agents actually to move away the object of interest.
    """

    def __init__(self, name):
        """Init method for the MoveAway behavior."""
        super(NewMoveAway, self).__init__(name)

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

        # Avoid Traps and Obstacles
        avoidto = AvoidTrapObstaclesBehaviour('MA_AVOID_3')
        avoidto.setup(0, self.agent)
        avoidto = FailureIsSuccess(avoidto)

        # Define move behavior
        move = Move('MA_MOVE_4')
        move.setup(0, self.agent)

        # Define a sequence to combine the primitive behavior
        mt_sequence = Sequence('MA_SEQUENCE')
        mt_sequence.add_children([goto, away, avoidto, move])

        self.behaviour_tree = BehaviourTree(mt_sequence)

    def initialise(self):
        """Everytime initialization. Not required for now."""
        pass

    def update(self):
        """Just call the tick method for the sequence.

        This will execute the primitive behaviors defined in the sequence
        """
        self.behaviour_tree.tick()
        return self.behaviour_tree.root.status
