"""Model class for single source foraging."""
from agent import SwarmAgentRandomSingleCarryDropReturn
from swarms.lib.model import Model
from swarms.lib.time import SimultaneousActivation
from swarms.lib.space import Grid
from swarms.lib.objects import Hub, Sites, Food
from swarms.utils.ui import UI


class SingleCarryDropReturnSwarmEnvironmentModel(Model):
    """A environment to model swarms."""

    def __init__(self, N, width, height, grid=10, seed=None, viewer=False):
        if seed is None:
            super(SingleCarryDropReturnSwarmEnvironmentModel, self).__init__(
                seed=None)
        else:
            super(SingleCarryDropReturnSwarmEnvironmentModel, self).__init__(
                seed)

        self.num_agents = N

        self.grid = Grid(width, height, grid)

        self.viewer = viewer

        self.schedule = SimultaneousActivation(self)

        self.hub = Hub(id=2, location=(0, 0), radius=5)
        self.grid.add_object_to_grid(self.hub.location, self.hub)
        self.site = Sites(id=3, location=(-35, -5), radius=5)
        self.grid.add_object_to_grid(self.site.location, self.site)

        self.agents = []
        self.foods = []

        for i in range(self.num_agents):
            f = Food(i, location=(-35, -5), radius=5)
            self.grid.add_object_to_grid(f.location, f)
            self.foods.append(f)

        for i in range(self.num_agents):
            a = SwarmAgentRandomSingleCarryDropReturn(i, self)
            self.schedule.add(a)
            x = -45
            y = -45
            a.location = (x, y)
            a.direction = -2.3561944901923448
            self.grid.add_object_to_grid((x, y), a)

            self.agents.append(a)

        if self.viewer:
            self.ui = UI(
                (width, height), [self.hub], self.agents, [self.site],
                food=self.foods)

    def step(self):
        self.schedule.step()
        if self.viewer:
            self.ui.step()
