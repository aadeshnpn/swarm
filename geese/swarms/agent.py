#from mesa.agent import Agent
from lib.agent import Agent

import random 

class SwarmAgent(Agent):
    """ An minimalistic swarm agent """
    def __init__(self, unique_id, model):
        super().__init__(unique_id,model)
        self.wealth = 1
        self.location = ()
        self.direction = np.random.random() * (2*np.pi)
        self.speed = 2
        
    def step(self):
        self.move()
        if self.wealth > 0:
            self.give_money()

    def advance(self):
        pass

    def move(self):
        self.location[0] = self.location[0] + np.cos(self.direction) * self.speed
        self.location[1] = self.location[1] + np.sin(self.direction) * self.speed
        self.location, self.direction = self.model.grid.check_limits(self.location, self.direction)        
        #possible_steps = self.model.grid.get_neighborhood(
        #    self.location,
        #    5
        #)

        #print (possible_steps)
        #new_position = random.choice(possible_steps)
        #self.model.grid.move_agent(self, new_position)

    def give_money(self):
        cellmates = self.model.grid.get_cell_list_contents([self.pos])
        if len(cellmates) > 1:
            other = random.choice (cellmates)
            other.wealth += 1
            self.wealth -= 1
