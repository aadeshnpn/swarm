from unittest import TestCase
from swarm.swarms.utils.bt import BTConstruct
import py_trees


class TestBT(TestCase):
    
    def setUp(self):
        self.bt = BTConstruct(None, xmlstring='<?xml version="1.0" encoding="UTF-8"?><Sequence><Sequence><Selector><cond>IsSingleCarry</cond><cond>IsMotionTrue</cond><act>RandomWalk</act></Selector><Sequence><cond>IsMoveable</cond><cond>IsMotionTrue</cond><act>GoTo</act></Sequence></Sequence><Sequence><Selector><cond>IsMotionTrue</cond><cond>IsMoveable</cond><cond>IsCarryable</cond><cond>IsMotionTrue</cond><act>RandomWalk</act></Selector><Sequence><cond>IsCarryable</cond><act>MultipleCarry</act></Sequence></Sequence></Sequence>')
        self.bt.construct()

    def test_agent_path(self):
        py_trees.logging.level = py_trees.logging.Level.DEBUG
        output = py_trees.display.ascii_tree(self.bt.behaviour_tree.root)
        output = output.replace('\n', '\k')
        self.maxDiff = None
        self.assertEqual('RootSequence\k[-] Sequence16\k    (-) Selector44\k        --> IsSingleCarry21\k        --> IsMotionTrue62\k        --> RandomWalk44\k    [-] Sequence23\k        --> IsMoveable14\k        --> IsMotionTrue58\k        --> GoTo78\k[-] Sequence81\k    (-) Selector52\k        --> IsMotionTrue53\k        --> IsMoveable16\k        --> IsCarryable30\k        --> IsMotionTrue27\k        --> RandomWalk53\k    [-] Sequence81\k        --> IsCarryable52\k        --> MultipleCarry41\k', output)