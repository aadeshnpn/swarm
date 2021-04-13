"""This is the mapper class which maps the xml file."""


import xml.etree.ElementTree as ET
import py_trees
from py_trees.composites import Sequence, Selector  # noqa: F401

from swarms.behaviors.scbehaviors import (      # noqa: F401
    MoveTowards, MoveAway, Explore, CompositeSingleCarry,
    CompositeMultipleCarry, CompositeDrop, CompositeDropCue,
    CompositePickCue, CompositeSendSignal, CompositeReceiveSignal,
    CompositeDropPartial, AvoidTrap, MakeAgentDead
    )
from swarms.behaviors.sbehaviors import (       # noqa: F401
    IsCarrying, NeighbourObjects, Move, IsDropable,
    IsVisitedBefore, IsInPartialAttached, IsAgentDead, IsPassable, IsDeathable
    )


class BTConstruct:
    """Mapper to map from xml to BT.

    This class maps xml file generated from grammar to
    Behavior Trees
    """

    def __init__(self, filename, agent, xmlstring=None):
        """Initialize the attributes for mapper.

        Args:
            filename: name of xml file that is to be mapped into BT
            agent: agent object
            xmlstring: xml stream instead of file
        """
        self.filename = filename
        self.xmlstring = xmlstring
        self.agent = agent

    def xmlfy(self):
        """Convert [] to <>."""
        self.xmlstring = self.xmlstring.replace('[', '<')
        self.xmlstring = self.xmlstring.replace(']', '>')
        self.xmlstring = self.xmlstring.replace('%', '"')

    def create_bt(self, root):
        """Recursive method to construct BT."""
        if len(list(root)) == 0:
            node_text = root.text
            # If the behavior needs to look for specific item
            if node_text.find('_') != -1:
                nodeval = node_text.split('_')
                # Check for behavior inversion
                if len(nodeval) == 2:
                    method, item = nodeval
                    behavior = eval(method)(method + str(
                        self.agent.model.random.randint(
                            100, 200)) + '_' + item)
                else:
                    method, item, _ = nodeval
                    behavior = py_trees.meta.inverter(eval(method))(
                        method + str(
                            self.agent.model.random.randint(
                                100, 200)) + '_' + item + '_inv')

                behavior.setup(0, self.agent, item)

            else:
                method = node_text
                behavior = eval(method)(method + str(
                    self.agent.model.random.randint(100, 200)))
                behavior.setup(0, self.agent, None)
            return behavior
        else:
            list1 = []
            for node in list(root):
                if node.tag not in ['cond', 'act']:
                    composits = eval(node.tag)(node.tag + str(
                        self.agent.model.random.randint(10, 90)))
                list1.append(self.create_bt(node))
                try:
                    if composits:
                        composits.add_children(list1.pop())
                        list1.append(composits)
                except (AttributeError, IndexError, UnboundLocalError) as e:
                    pass

            return list1

    def construct(self):
        """Create a tree from xml."""
        if self.xmlstring is not None:
            self.xmlfy()
            tree = ET.fromstring(self.xmlstring)
            self.root = tree

        elif self.filename is not None:
            tree = ET.parse(self.filename)
            self.root = tree.getroot()
        else:
            print("Cannont create BT. Check the filename or stream")
            exit()

        whole_list = self.create_bt(self.root)
        top = eval(self.root.tag)('Root' + self.root.tag)
        top.add_children(whole_list)
        self.behaviour_tree = py_trees.trees.BehaviourTree(top)
        # py_trees.logging.level = py_trees.logging.Level.DEBUG
        # py_trees.display.print_ascii_tree(top)

    def visualize(self, name='bt.png'):
        """Save bt graph to a file."""
        py_trees.display.render_dot_tree(self.behaviour_tree.root, name=name)
