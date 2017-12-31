import sys, os

sys.path += [os.path.join('/home/aadeshnpn/Documents/BYU/hcmi/swarm')]

#from mesa import Agent, Model
#from mesa.time import SimultaneousActivation, RandomActivation
#from mesa.space import MultiGrid

#from lib.agent import Agent
from lib.model import Model
from lib.time import SimultaneousActivation, RandomActivation
from lib.space import Grid

from swarms.agent import SwarmAgent

class EnvironmentModel(Model):
    """ A environemnt to model swarms """
    def __init__(self, N, width, height):
        super().__init__()                
        self.num_agents = N
        #self.grid = MultiGrid (width, height, True)
        self.grid = Grid (50, 50, 5)        
        self.schedule = SimultaneousActivation(self)
        #self.schedule = RandomActivation(self)  

        for i in range(self.num_agents):
            a = SwarmAgent(i, self) 
            self.schedule.add(a)

            # Add the agent to a random grid cell
            x = self.random.randint(-self.grid.width/2, self.grid.width/2)
            y = self.random.randint(-self.grid.height/2,self.grid.height/2)

            #grid_key, grid_value = self.grid.find_grid((x,y))
            self.grid.add_object_to_grid((x,y), a)

            #self.grid.place_agent(a,(x,y))

    def step(self):
        self.schedule.step()



#if __name__ == '__main__':
#   main()