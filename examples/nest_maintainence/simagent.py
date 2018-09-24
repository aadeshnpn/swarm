"""Derived agent class."""

from swarms.lib.agent import Agent
import numpy as np
from swarms.utils.bt import BTConstruct
from swarms.behaviors.scbehaviors import Move
# from swarms.utils.results import Results
from py_trees import Status
import copy


# Define a dummy behavior for the move.
class Dummymove(Move):
    """Dummy move behavior."""

    def __init__(self, name):
        super(Dummymove, self).__init__(name)

    def setup(self, timeout, agent, item=None):
        """Setup."""
        pass

    def initialise(self):
        """Pass."""
        pass

    def update(self):
        return Status.SUCCESS


class SimAgent(Agent):
    """Simulation agent.

    An minimalistic behavior tree for swarm agent
    implementing carry and drop behavior.
    """

    def __init__(self, name, model, xmlstring=None):
        super().__init__(name, model)
        self.location = ()

        self.direction = model.random.rand() * (2 * np.pi)
        self.speed = 2
        self.radius = 3

        self.moveable = True
        self.shared_content = dict()

        self.carryable = False
        # Define a BTContruct object
        self.bt = BTConstruct(None, self)

        class DummyIndividual:
            def __init__(self):
                self.phenotype = None
        dummyind = DummyIndividual()
        self.individual = [dummyind]
        self.individual[0].phenotype = xmlstring

        self.bt.xmlstring = xmlstring
        self.bt.construct()

    def step(self):
        self.bt.behaviour_tree.tick()

    def advance(self):
        pass

    def replace_nodes(self):
        dummy_bt = copy.copy(self.bt)
        # dummy_bt.behaviour_tree.tick()
        root = dummy_bt.behaviour_tree.root

        # For now dummpy node is move but it could be different
        name = 'Dummy' + str(self.model.random.randint(0, 1000, 1)[0])
        dummynode = Dummymove(name)

        def replace(roots, node):
            if type(node).__name__ == 'Move':
                roots.replace_child(node, dummynode)

        for node in root.iterate():
            try:
                innerroot = node.behaviour_tree.root
                for innernode in innerroot.iterate():
                    replace(innerroot, innernode)
            except AttributeError:
                replace(root, node)

        return dummy_bt


class SimAgentRes1(SimAgent):
    """Simulation agent for resilience.

    An minimalistic behavior tree for swarm agent
    to simulate resilient behaviors when actuator is not
    working.
    """

    def __init__(self, name, model, xmlstring=None):
        super().__init__(name, model, xmlstring)

        # When the actuator is not working. We repalce the move
        # premitive behavior with dummpy move behaviors 50% of
        # the time.
        self.dummy_bt = self.replace_nodes()

        # Move node only works 50% of the time
        # py_trees.logging.level = py_trees.logging.Level.DEBUG
        # py_trees.display.print_ascii_tree(self.dummy_bt.behaviour_tree.root)
        """
        for a in self.dummy_bt.behaviour_tree.root.iterate():
            try:
                py_trees.display.print_ascii_tree(a.behaviour_tree.root)
            except AttributeError:
                pass
        """

    def step(self):
        # Only the actuators work 50% of the time
        if self.model.random.rand() > 0.5:
            self.bt.behaviour_tree.tick()
        else:
            self.dummy_bt.behaviour_tree.tick()

    def replace_nodes(self):
        dummy_bt = copy.copy(self.bt)
        # dummy_bt.behaviour_tree.tick()
        root = dummy_bt.behaviour_tree.root

        # For now dummpy node is move but it could be different
        name = 'Dummy' + str(self.model.random.randint(0, 1000, 1)[0])
        dummynode = Dummymove(name)

        def replace(roots, node):
            if type(node).__name__ == 'Move':
                roots.replace_child(node, dummynode)

        for node in root.iterate():
            try:
                innerroot = node.behaviour_tree.root
                for innernode in innerroot.iterate():
                    replace(innerroot, innernode)
            except AttributeError:
                replace(root, node)

        return dummy_bt


class SimAgentRes2(SimAgent):
    """Simulation agent for resilience.

    An minimalistic behavior tree for swarm agent
    to simulate resilient behaviors a node is removed.
    """

    def __init__(self, name, model, xmlstring=None):
        super().__init__(name, model, xmlstring)

        # When the actuator is not working. We repalce the move
        # premitive behavior with dummpy move behaviors 50% of
        # the time.
        self.dummy_bt = self.replace_nodes()

        # Move node only works 50% of the time
        # py_trees.logging.level = py_trees.logging.Level.DEBUG
        # py_trees.display.print_ascii_tree(self.dummy_bt.behaviour_tree.root)
        """
        for a in self.dummy_bt.behaviour_tree.root.iterate():
            try:
                py_trees.display.print_ascii_tree(a.behaviour_tree.root)
            except AttributeError:
                pass
        """

    def step(self):
        # Only the actuators work 50% of the time
        if self.model.random.rand() > 0.5:
            self.bt.behaviour_tree.tick()
        else:
            self.dummy_bt.behaviour_tree.tick()

    def replace_nodes(self):
        dummy_bt = copy.copy(self.bt)
        root = dummy_bt.behaviour_tree.root

        # For now dummpy node is removed but it could be different
        def remove(roots, node):
            if type(node).__name__ == 'Move':
                roots.remove_child(node)

        for node in root.iterate():
            try:
                innerroot = node.behaviour_tree.root
                for innernode in innerroot.iterate():
                    remove(innerroot, innernode)
            except AttributeError:
                remove(root, node)

        return dummy_bt


class SimAgentResComm1(SimAgent):
    """Simulation agent for resilience.

    An minimalistic behavior tree for swarm agent
    to simulate resilient behaviors when communication
    hardware is zammed
    """

    def __init__(self, name, model, xmlstring=None):
        super().__init__(name, model, xmlstring)

        # When the actuator is not working. We repalce the move
        # premitive behavior with dummpy move behaviors 50% of
        # the time.
        self.dummy_bt = self.replace_nodes()

        # Move node only works 50% of the time
        # py_trees.logging.level = py_trees.logging.Level.DEBUG
        # py_trees.display.print_ascii_tree(self.dummy_bt.behaviour_tree.root)
        """
        for a in self.dummy_bt.behaviour_tree.root.iterate():
            try:
                py_trees.display.print_ascii_tree(a.behaviour_tree.root)
            except AttributeError:
                pass
        """

    def step(self):
        # Only the actuators work 50% of the time
        if self.model.random.rand() > 0.5:
            self.bt.behaviour_tree.tick()
        else:
            self.dummy_bt.behaviour_tree.tick()

    def replace_nodes(self):
        dummy_bt = copy.copy(self.bt)
        # dummy_bt.behaviour_tree.tick()
        root = dummy_bt.behaviour_tree.root

        # For now dummpy node is move but it could be different
        name = 'Dummy' + str(self.model.random.randint(0, 1000, 1)[0])
        dummynode = Dummymove(name)

        def replace(roots, node):
            communication_list = [
                'SendSignal', 'ReceiveSignal', 'DropCue', 'PickCue']
            if type(node).__name__ in communication_list:
                roots.replace_child(node, dummynode)

        for node in root.iterate():
            try:
                innerroot = node.behaviour_tree.root
                for innernode in innerroot.iterate():
                    replace(innerroot, innernode)
            except AttributeError:
                replace(root, node)

        return dummy_bt


class SimAgentResComm2(SimAgent):
    """Simulation agent for resilience.

    An minimalistic behavior tree for swarm agent
    to simulate resilient behaviors a node is removed.
    """

    def __init__(self, name, model, xmlstring=None):
        super().__init__(name, model, xmlstring)

        # When the actuator is not working. We repalce the move
        # premitive behavior with dummpy move behaviors 50% of
        # the time.
        self.dummy_bt = self.replace_nodes()

        # Move node only works 50% of the time
        # py_trees.logging.level = py_trees.logging.Level.DEBUG
        # py_trees.display.print_ascii_tree(self.dummy_bt.behaviour_tree.root)
        """
        for a in self.dummy_bt.behaviour_tree.root.iterate():
            try:
                py_trees.display.print_ascii_tree(a.behaviour_tree.root)
            except AttributeError:
                pass
        """

    def step(self):
        # Only the actuators work 50% of the time
        if self.model.random.rand() > 0.5:
            self.bt.behaviour_tree.tick()
        else:
            self.dummy_bt.behaviour_tree.tick()

    def replace_nodes(self):
        dummy_bt = copy.copy(self.bt)
        root = dummy_bt.behaviour_tree.root

        # For now dummpy node is removed but it could be different
        def remove(roots, node):
            communication_list = [
                'SendSignal', 'ReceiveSignal', 'DropCue', 'PickCue']
            if type(node).__name__ in communication_list:
                roots.remove_child(node)

        for node in root.iterate():
            try:
                innerroot = node.behaviour_tree.root
                for innernode in innerroot.iterate():
                    remove(innerroot, innernode)
            except AttributeError:
                remove(root, node)

        return dummy_bt
