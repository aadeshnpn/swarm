from swarms.lib.model import Model
from swarms.lib.time import SimultaneousActivation
from swarms.lib.space import Grid
from swarms.lib.objects import Obstacles, Sites, Hub, Traps
import numpy as np

from agent import SwarmAgent
# Global variables for width and height
width = 500
height = 500


class WealthEnvironmentModel(Model):
    """ A environemnt to model swarms """
    def __init__(self, N, width, height, grid=10, seed=None):
        if seed is None:
            super(WealthEnvironmentModel, self).__init__(seed=None)
        else:
            super(WealthEnvironmentModel, self).__init__(seed)

        self.num_agents = N

        self.grid = Grid(width, height, grid)

        self.schedule = SimultaneousActivation(self)

        self.hub = Hub(id=1, location=(+145, -145), radius=5)
        self.grid.add_object_to_grid(self.hub.location, self.hub)

        self.site = Sites(id=2, location=(155, 155), radius=15, q_value=0.9)
        self.grid.add_object_to_grid(self.site.location, self.site)

        self.obstacles = Obstacles(id=3, location=(40, 80), radius=29)
        self.grid.add_object_to_grid(self.obstacles.location, self.obstacles)

        self.traps = Traps(id=4, location=(-40, -40), radius=21)
        self.grid.add_object_to_grid(self.traps.location, self.traps)

        self.agents = []
        for i in range(self.num_agents):
            a = SwarmAgent(i, self)
            self.schedule.add(a)
            # Add the agent to a random grid cell
            x = self.random.randint(-self.grid.width / 2, self.grid.width / 2)
            y = self.random.randint(-self.grid.height / 2, self.grid.height / 2)

            a.location = (x, y)
            self.grid.add_object_to_grid((x, y), a)
            self.agents.append(a)

    def step(self):
        self.schedule.step()