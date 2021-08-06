from swarms.lib.model import Model
from swarms.lib.space import Grid
from swarms.lib.objects import Obstacles, Sites, Hub, Traps
from swarms.lib.time import SimultaneousActivation

from agent import SwarmAgentAvoid


class SimTrapModel(Model):
    """ A environemnt to model swarms """
    def __init__(self, N, width, height, grid=10, seed=None):
        if seed is None:
            super(SimTrapModel, self).__init__(seed=None)
        else:
            super(SimTrapModel, self).__init__(seed)

        self.num_agents = N

        self.grid = Grid(width, height, grid)

        self.schedule = SimultaneousActivation(self)

        self.hub = Hub(id=1, location=(+145, -145), radius=5)
        self.grid.add_object_to_grid(self.hub.location, self.hub)

        self.target = Sites(id=2, location=(155, 155), radius=15, q_value=0.9)
        self.grid.add_object_to_grid(self.target.location, self.target)

        self.obstacle1 = Obstacles(id=3, location=(40, 80), radius=29)
        self.grid.add_object_to_grid(self.obstacle1.location, self.obstacle1)

        self.obstacle2 = Obstacles(id=4, location=(100, 15), radius=20)
        self.grid.add_object_to_grid(self.obstacle2.location, self.obstacle2)

        self.obstacles = [self.obstacle1, self.obstacle2]

        self.trap1 = Traps(id=5, location=(-40, -40), radius=21)
        self.grid.add_object_to_grid(self.trap1.location, self.trap1)

        self.trap2 = Traps(id=6, location=(100, 80), radius=20)
        self.grid.add_object_to_grid(self.trap2.location, self.trap2)

        self.traps = [self.trap1, self.trap2]

        self.agents = []
        for i in range(self.num_agents):
            a = SwarmAgentAvoid(i, self)
            self.schedule.add(a)
            x = -190 + self.random.randint(-10, 70)
            y = -190 + self.random.randint(-10, 70)
            a.location = (x, y)
            a.direction = -2.3661944901923448
            self.grid.add_object_to_grid((x, y), a)
            self.agents.append(a)

        self.agent = a
        self.agents

    def step(self):
        self.schedule.step()
