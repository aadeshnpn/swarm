from lib.agent import Agent

import numpy as np


class SwarmAgent(Agent):
    """ An minimalistic swarm agent """
    def __init__(self, name, model):
        super().__init__(name, model)
        self.location = ()

        self.direction = model.random.rand() * (2 * np.pi)
        self.speed = 2
        self.radius = 3
        self.wealth = 1

    # Make necessary Changes
    def step(self):
        if self.wealth > 0:
            self.give_money()

    # Applies the changes
    def advance(self):
        self.move()

    def move(self):
        new_location = ()
        x = int(self.location[0] + np.cos(self.direction) * self.speed)
        y = int(self.location[1] + np.sin(self.direction) * self.speed)
        new_location, direction = self.model.grid.check_limits((x, y), self.direction)
        self.model.grid.move_object(self.location, self, new_location)
        self.location = new_location
        self.direction = direction

    def give_money(self):
        cellmates = self.model.grid.get_objects_from_grid('SwarmAgent', self.location)

        if len(cellmates) > 1:
            other = self.model.random.choice(cellmates)
            other.wealth += 1
            self.wealth -= 1
