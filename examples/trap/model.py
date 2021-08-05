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

        self.target = Sites(id=2, location=(145, 145), radius=5, q_value=0.5)
        self.grid.add_object_to_grid(self.target.location, self.target)

        self.obstacles = Obstacles(id=3, location=(0, 0), radius=38)
        self.grid.add_object_to_grid(self.obstacles.location, self.obstacles)

        self.trap = Traps(id=4, location=(110, 130), radius=8)
        self.grid.add_object_to_grid(self.trap.location, self.trap)

        self.agents = []
        for i in range(self.num_agents):
            a = SwarmAgentAvoid(i, self)
            self.schedule.add(a)
            x = -80 #-190 + self.random.randint(-10, 50)
            y = -80 #-190 + self.random.randint(-10, 50)
            a.location = (x, y)
            a.direction = -2.3561944901923448
            self.grid.add_object_to_grid((x, y), a)
            self.agents.append(a)

        self.agent = a
        self.agents

    def step(self):
        self.schedule.step()
