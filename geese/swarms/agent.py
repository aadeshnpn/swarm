#from mesa.agent import Agent
from lib.agent import Agent

import numpy as np

class SwarmAgent(Agent):
    """ An minimalistic swarm agent """
    def __init__(self, unique_id, model):
        super().__init__(unique_id,model)
        self.wealth = 1
        self.location = ()

        self.direction = model.random.rand() * (2*np.pi)
        self.speed = 2
        
    def step(self):
        #self.move()
        if self.wealth > 0:
            self.give_money()

    def advance(self):
        self.move()

    def move(self):
        new_location = ()
        x = self.location[0] + np.cos(self.direction) * self.speed
        y = self.location[1] + np.sin(self.direction) * self.speed
        new_location, direction = self.model.grid.check_limits((x,y), self.direction)        
        self.location = new_location
        self.direction = direction


    def give_money(self):
        #cellmates = self.model.grid.get_cell_list_contents([self.pos])#
        cellmates = self.model.grid.get_objects('SwarmAgent', self.location)

        if len(cellmates) > 1:
            other = self.model.random.choice (cellmates)
            other.wealth += 1
            self.wealth -= 1
