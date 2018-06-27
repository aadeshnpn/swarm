# -*- coding: utf-8 -*-
"""
The agent class for swarm framework.

Core Objects: Agent

"""


class Agent:
    """Base class for a agent."""

    def __init__(self, name, model):
        """Create a new agent.

        Overload this method to define diverse agents.
        Args:
            name: a unique name for the agent. It helps to
                  distingush between different agents in the environment.

            model: model class which gives the agent access to environmental
                    variables like sites, hub, food and others

        Attributes:
            weight: agent's weight

            capacity: agent's capacity to do some work in the environment

            attached_objects: a list which stores the objects attached
                               to the agent. Useful for carrying and droping
                               objects

            partial_attached_objects: a list which stores the objects
                                    partially attached to the agent. Useful
                                    for multi-carry behaviors

        """
        self.name = name
        self.model = model
        self.weight = 5
        self.capacity = self.weight * 2
        self.force = 10.0
        self.attached_objects = []
        self.partial_attached_objects = []
        self.signals = []

    def step(self):
        """Represent a single step of the agent."""
        pass

    def advance(self):
        """Actions to do after a step of the agent."""
        pass

    def get_weight(self):
        """Compute the weight of the agent.

        The weight of the agent changes based on the objects it
        carries. We take the weight of partially attached objects as well
        as fully carrying objects while computing the weight of the
        agent.
        """
        relative_weight = self.weight
        for item in self.attached_objects:
            try:
                relative_weight += item.weight
            except AttributeError:
                pass
        try:
            relative_weight += self.partial_attached_objects[0].weight
        except (AttributeError, IndexError):
            pass

        return relative_weight

    def get_capacity(self):
        """Compute the capacity of the agent.

        The capacity of the agent is fixed. Based on the objects it is
        carrying we need to adjust/reflect on the capacity of the agent.
        """
        relative_capacity = self.capacity
        for item in self.attached_objects:
            try:
                relative_capacity -= item.agents[self]
            except KeyError:
                self.attached_objects.remove(item)

        for item in self.partial_attached_objects:
            try:
                relative_capacity -= item.agents[self]
            except KeyError:
                self.partial_attached_objects.remove(item)

        if relative_capacity < 0:
            return 0
        else:
            return relative_capacity
