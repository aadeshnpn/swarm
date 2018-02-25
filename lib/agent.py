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
        self.capacity = self.weight * 2
        self.force = 10.0
        self.attached_objects = []
        self.partial_attached_objects = []

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
        try:
            relative_weight += self.partial_attached_objects.weight
        except AttributeError:
            pass              

        return relative_weight

    """
    def bound_capacity(self, rc, weight):
        if rc - weight >= 0:
            rc -= weight
        else:
            rc = 0
        return rc

    def get_capacity(self):
        relative_capacity = self.capacity
        for item in self.attached_objects:
            try:
                if relative_capacity > 0:
                    relative_capacity = self.bound_capacity(
                        relative_capacity, item.weight)
                else:
                    item.agents.remove(self)
                    self.attached_objects.remove(item)
            except AttributeError:
                pass

        for item in self.partial_attached_objects:
            try:
                if relative_capacity > 0:
                    relative_capacity = self.bound_capacity(
                        relative_capacity, item.weight)
                else:
                    item.agents.remove(self)
                    self.partial_attached_objects.remove(item)
            except AttributeError:
                pass                
            print ('get capacity', relative_capacity, item.weight)                
        return relative_capacity
    """

    def get_capacity(self):
        relative_capacity = self.capacity
        for item in self.attached_objects:
            # indx = item.agents.index(self)
            relative_capacity -= item.agents[self]

        for item in self.partial_attached_objects:
            # indx = item.agents.index(self)
            relative_capacity -= item.agents[self]

        if relative_capacity < 0:
            return 0
        else:
            return relative_capacity
