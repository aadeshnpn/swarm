#import sys, os

#sys.path += [os.path.join('/home/aadeshnpn/Documents/BYU/hcmi/swarm')]

from lib.model import Model
from lib.time import SimultaneousActivation, RandomActivation, StagedActivation
from lib.space import Grid

from swarms.agent import SwarmAgent

class EnvironmentModel(Model):
    """ A environemnt to model swarms """
    def __init__(self, N, width, height, grid=10, seed=None):
        if seed == None:
            super().__init__()
        else:
            super().__init__(seed)            
            
        self.num_agents = N

        self.grid = Grid (width, height, grid) 
    
        self.schedule = SimultaneousActivation(self)

        for i in range(self.num_agents):
            a = SwarmAgent(i, self) 
            self.schedule.add(a)

            # Add the agent to a random grid cell
            x = self.random.randint(-self.grid.width/2, self.grid.width/2)
            y = self.random.randint(-self.grid.height/2, self.grid.height/2)

            self.grid.add_object_to_grid((x,y), a)

    def step(self):
        self.schedule.step()

