from py_trees import Behaviour, Status
import numpy as np
from swarms.utils.distangle import get_direction


# Defining behaviors for the agent

# Sense behavior for the agent
class NeighbourObjects(Behaviour):
    def __init__(self, name):
        super(NeighbourObjects, self).__init__(name)

    def setup(self, timeout, agent, object_name):
        self.agent = agent
        self.object_name = object_name

    def initialise(self):
        pass

    def update(self):
        grids = self.agent.model.grid.get_neighbourhood(self.agent.location, self.radius)
        objects = self.agent.model.grid.get_objects_from_list_of_grid(self.object_name, grids)
        if len(objects) > 1:
            self.agent.shared_content[self.object_name] = objects
            return Status.SUCCESS
        else:
            self.agent.shared_content[self.object_name] = []
            return Status.FAILURE


# Behaviors defined for move
class GoTo(Behaviour):
    def __init__(self, name):
        super(GoTo, self).__init__(name)

    def setup(self, timeout, agent, thing):
        self.agent = agent
        self.thing = thing

    def initialise(self):
        pass

    def update(self):
        self.agent.direction = get_direction(self.thing.location, self.agent.location)
        return Status.SUCCESS


# Behaviors defined for move
class RandomWalk(Behaviour):
    def __init__(self, name):
        super(GoTo, self).__init__(name)

    def setup(self, timeout, agent):
        self.agent = agent

    def initialise(self):
        pass

    def update(self):
        delta_d = self.agent.model.random.normal(0, .1)
        self.agent.direction = (self.agent.direction + delta_d) % (2 * np.pi)
        return Status.SUCCESS


class Move(Behaviour):
    def __init__(self, name):
        super(Move, self).__init__(name)

    def setup(self, timeout, agent):
        self.agent = agent

    def initialise(self):
        pass

    def update(self):
        x = int(self.agent.location[0] + np.cos(self.agent.direction) * self.agent.speed)
        y = int(self.agent.location[1] + np.sin(self.agent.direction) * self.agent.speed)
        new_location, direction = self.agent.model.grid.check_limits((x, y), self.agent.direction)
        self.agent.model.grid.move_object(self.agent.location, self.agent, new_location)
        self.agent.location = new_location
        self.agent.direction = direction
        return Status.SUCCESS


class DoNotMove(Behaviour):
    def __init__(self, name):
        super(Move, self).__init__(name)

    def setup(self, timeout, agent):
        self.agent = agent

    def initialise(self):
        pass

    def update(self):
        return Status.SUCCESS
