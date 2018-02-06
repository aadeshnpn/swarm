# -*- coding: utf-8 -*-
"""
The agent class for swarm framework.

Core Objects: Agent

"""


class Agent:
    """ Base class for a agent. """
    def __init__(self, name, model):
        """ Create a new agent. """
        self.name = name
        self.model = model
        self.weight = 5
        self.force = 10.0
        self.attached_objects = []

    def step(self):
        """ A single step of the agent. """
        pass

    def advance(self):
        pass

    def get_weight(self):
        relative_weight = self.weight
        for item in self.attached_objects:
            try:
                relative_weight += item.weight
            except AttributeError:
                pass
        return relative_weight