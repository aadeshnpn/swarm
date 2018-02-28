"""This is the mapper which maps the xml file."""

import xml.etree.ElementTree as ET
import py_trees 
from py_trees.composites import Sequence, Selector
import random

from swarms.sbehaviors import (
    IsCarryable, IsSingleCarry, SingleCarry,
    NeighbourObjects, IsMultipleCarry, IsInPartialAttached,
    InitiateMultipleCarry, IsEnoughStrengthToCarry,
    Move, GoTo, IsMotionTrue, RandomWalk, IsMoveable,
    MultipleCarry
    )


def create_bt(root):
    if len(list(root)) == 0:
        behavior = eval(root.text)(root.text + str(random.randint(10, 90)))
        return behavior
    else:
        list1 = []
        for node in list(root):
            if node.tag not in ['cond', 'act']:
                composits = eval(node.tag)(node.tag + str(random.randint(10, 90)))
            list1.append(create_bt(node))
            try:
                if composits:
                    composits.add_children(list1.pop())
                    list1.append(composits)
            except:
                pass

        return list1


class ConstructBT:
    """Mapper to map from xml to BT."""

    """This class maps xml file generated from grammar to
    Behavior Trees"""    
    def __init__(self, filename, xmlstring=None):
        self.filename = filename
        self.xmlstring = xmlstring

    def construct(self):
        if self.xmlstring is not None:
            tree = ET.fromstring(self.xmlstring)
            self.root = tree
        elif self.filename is not None:
            tree = ET.parse(self.filename)
            self.root = tree.getroot()
        else:
            print("Cannont create BT. Check the filename or stream")
            exit()

        whole_list = create_bt(self.root)
        top = eval(self.root.tag)('Root' + self.root.tag)
        top.add_children(whole_list)
        self.behaviour_tree = py_trees.trees.BehaviourTree(top)
        #py_trees.logging.level = py_trees.logging.Level.DEBUG
        #py_trees.display.print_ascii_tree(top)            


def main():
    bt = ConstructBT("/home/aadeshnpn/Documents/BYU/hcmi/swarm/swarms/utils/bt.xml")
    #bt = ConstructBT(None, xmlstring='<?xml version="1.0" encoding="UTF-8"?><Sequence><Sequence><Selector><cond>IsSingleCarry</cond><cond>IsMotionTrue</cond><act>RandomWalk</act></Selector><Sequence><cond>IsMoveable</cond><cond>IsMotionTrue</cond><act>GoTo</act></Sequence></Sequence><Sequence><Selector><cond>IsMotionTrue</cond><cond>IsMoveable</cond><cond>IsCarryable</cond><cond>IsMotionTrue</cond><act>RandomWalk</act></Selector><Sequence><cond>IsCarryable</cond><act>MultipleCarry</act></Sequence></Sequence></Sequence>')
    bt.construct()


if __name__ == '__main__':
    main()

