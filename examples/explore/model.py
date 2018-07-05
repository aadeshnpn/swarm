"""A simple environmen model."""

from agent import SwarmAgentRandomWalk
from swarms.lib.model import Model
from swarms.lib.time import SimultaneousActivation
from swarms.lib.space import Grid

from swarms.lib.objects import Sites, Hub
from swarms.utils.ui import UI


class RandomWalkSwarmEnvironmentModel(Model):
    """A environment to model swarms."""

    def __init__(self, N, width, height, grid=10, seed=None, viewer=False):
        """Initialize the environment methods."""
        if seed is None:
            super(RandomWalkSwarmEnvironmentModel, self).__init__(
                seed=None)
        else:
            super(RandomWalkSwarmEnvironmentModel, self).__init__(
                seed)

        self.num_agents = N

        self.grid = Grid(width, height, grid)

        self.viewer = viewer

        self.schedule = SimultaneousActivation(self)

        self.hub = Hub(id=2, location=(0, 0), radius=5)
        self.grid.add_object_to_grid(self.hub.location, self.hub)

        self.site = Sites(id=2, location=(25, 25), radius=5)
        self.grid.add_object_to_grid(self.site.location, self.site)

        self.agents = []
        for i in range(self.num_agents):
            a = SwarmAgentRandomWalk(i, self)
            self.schedule.add(a)
            x = 0
            y = 0
            a.location = (x, y)
            a.direction = -2.3561944901923448
            self.grid.add_object_to_grid((x, y), a)
            self.agents.append(a)

        if self.viewer:
            self.ui = UI((width, height), [self.hub], self.agents, [self.site])

    def step(self):
        """Execute."""
        self.schedule.step()
        if self.viewer:
            self.ui.step()
