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

    def step(self):
        """ A single step of the agent. """
        pass
