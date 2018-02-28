from unittest import TestCase
from swarm.swarms.utils.bt import BTConstruct
import py_trees

class TestBT(TestCase):
    
    def setUp(self):
        self.bt = BTConstruct(None, xmlstring='<?xml version="1.0" encoding="UTF-8"?><Sequence><Sequence><Selector><cond>IsSingleCarry</cond><cond>IsMotionTrue</cond><act>RandomWalk</act></Selector><Sequence><cond>IsMoveable</cond><cond>IsMotionTrue</cond><act>GoTo</act></Sequence></Sequence><Sequence><Selector><cond>IsMotionTrue</cond><cond>IsMoveable</cond><cond>IsCarryable</cond><cond>IsMotionTrue</cond><act>RandomWalk</act></Selector><Sequence><cond>IsCarryable</cond><act>MultipleCarry</act></Sequence></Sequence></Sequence>')
        self.bt.construct()

    def test_agent_path(self):
        py_trees.logging.level = py_trees.logging.Level.DEBUG
        py_trees.display.print_ascii_tree(self.bt.behaviour_tree)   
        self.assertEqual((5, 5), (27, 27))