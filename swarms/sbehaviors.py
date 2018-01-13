from py_trees import Behaviour, Status
import numpy as np


class HasMoney(Behaviour):
    def __init__(self, name):
        super(HasMoney, self).__init__(name)

    def setup(self, timeout, agent):
        self.agent = agent

    def initialise(self):
        pass

    def update(self):
        if self.agent.wealth > 0:
            return Status.SUCCESS
        else:
            return Status.FAILURE

    #def terminate(self):
    #    pass


class NeighbourCondition(Behaviour):
    def __init__(self, name):
        super(NeighbourCondition, self).__init__(name)

    def setup(self, timeout, agent):
        self.agent = agent

    def initialise(self):
        pass

    def update(self):
        cellmates = self.agent.model.grid.get_objects_from_grid('SwarmAgent', self.agent.location)
        if len(cellmates) > 1:
            return Status.SUCCESS
        else:
            return Status.FAILURE

    #def terminate(self):
    #    pass


class ShareMoney(Behaviour):
    def __init__(self, name):
        super(ShareMoney, self).__init__(name)

    def setup(self, timeout, agent):
        self.agent = agent

    def initialise(self):
        pass

    def update(self):
        try:
            cellmates = self.agent.model.grid.get_objects_from_grid('SwarmAgent', self.agent.location)
            others = self.agent.model.random.choice(cellmates)
            others.wealth += 1
            self.agent.wealth -= 1
            return Status.SUCCESS
        except:
            return Status.FAILURE

    #def terminate(self):
    #    pass


class Move(Behaviour):
    def __init__(self, name):
        super(Move, self).__init__(name)

    def setup(self, timeout, agent):
        self.agent = agent

    def initialise(self):
        pass

    def update(self):
        try:
            x = int(self.agent.location[0] + np.cos(self.agent.direction) * self.agent.speed)
            y = int(self.agent.location[1] + np.sin(self.agent.direction) * self.agent.speed)
            new_location, direction = self.agent.model.grid.check_limits((x, y), self.agent.direction)
            self.agent.model.grid.move_object(self.agent.location, self.agent, new_location)
            self.agent.location = new_location
            self.agent.direction = direction
            return Status.SUCCESS
        except:
            return Status.FAILURE

    #def terminate(self):
    #    pass
