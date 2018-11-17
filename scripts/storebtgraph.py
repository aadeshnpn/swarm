"""Script to store BT graph."""

import py_trees

from swarms.utils.jsonhandler import JsonPhenotypeData
from swarms.utils.bt import BTConstruct
from swarms.lib.agent import Agent
from swarms.lib.model import Model

from swarms.behaviors.scbehaviors import (      # noqa: F401
    MoveTowards, MoveAway, Explore, CompositeSingleCarry,
    CompositeMultipleCarry, CompositeDrop, CompositeDropCue,
    CompositePickCue, CompositeSendSignal, CompositeReceiveSignal,
    CompositeDropPartial
    )


def main():
    """Parse args and call bt visualize module."""
    # dname = '/home/aadeshnpn/Documents/BYU/hcmi/hri/supporting_materials/nest_maintenance'    # noqa: E501
    # jfname = dname + '/1539014820252.json'
    # dname = '/home/aadeshnpn/Documents/BYU/hcmi/hri/supporting_materials/cooperative_transport'   # noqa: E501
    # jfname = dname + '/1538447335350.json'
    dname = '/home/aadeshnpn/Documents/BYU/hcmi/hri/supporting_materials/foraging'  # noqa: E501
    jfname = dname + '/1538473090382007.json'
    jdata = JsonPhenotypeData.load_json_file(jfname)
    phenotypes = jdata['phenotypes']
    for i in range(len(phenotypes)):
        # if i >= 5:
        #    break
        m = Model()
        a = Agent('1', m)
        bt = BTConstruct(None, a, phenotypes[i])
        bt.construct()
        bt.visualize(name=dname + '/' + str(i))


def gen_from_xmlstring():
    """Generate BT graph from xml string."""
    dname = '/home/aadeshnpn/Documents/BYU/hcmi/hri/supporting_materials/nest_maintenance'  # noqa: E501
    xmlstring = ''
    m = Model()
    a = Agent('1', m)
    bt = BTConstruct(None, a, xmlstring)
    bt.construct()
    bt.visualize(name=dname + '/' + str(90))


def compositebehaviors():
    """Generate BT graph from composite behaviors."""
    behaviors = [
        MoveTowards, MoveAway, Explore, CompositeSingleCarry,
        CompositeMultipleCarry, CompositeDrop, CompositeDropCue,
        CompositePickCue, CompositeSendSignal, CompositeReceiveSignal,
        CompositeDropPartial]

    dname = '/home/aadeshnpn/Documents/BYU/hcmi/hri/supporting_materials/composite_behaviors'   # noqa: E501

    for i in range(len(behaviors)):
        behavior = behaviors[i](str(i))
        behavior.setup(0, None, None)
        # print (behavior.behaviour_tree.root)
        py_trees.display.render_dot_tree(
            behavior.behaviour_tree.root,
            visibility_level=py_trees.common.VisibilityLevel.ALL,
            name=dname + '/' + str(i))
        # print (py_trees.display.ascii_tree(behavior.behaviour_tree.root))


if __name__ == '__main__':
    main()
    # compositebehaviors()
    # gen_from_xmlstring()
