from py_trees import Behaviour, Status, meta, composites, Blackboard
import numpy as np
from swarms.utils.distangle import get_direction


# Defining behaviors for the agent

# Sense behavior for the agent update using blackboard
class NeighbourObjects(Behaviour):
    def __init__(self, name):
        super(NeighbourObjects, self).__init__(name)
        self.blackboard = Blackboard()

    def setup(self, timeout, agent, object_name):
        self.agent = agent
        self.object_name = object_name
        self.blackboard.shared_content = dict()

    def initialise(self):
        pass

    def update(self):
        grids = self.agent.model.grid.get_neighborhood(self.agent.location, self.agent.radius)
        objects = self.agent.model.grid.get_objects_from_list_of_grid(self.object_name, grids)
        if len(objects) >= 1:
            self.blackboard.shared_content[self.object_name] = objects
            self.agent.shared_content[self.object_name] = objects
            return Status.SUCCESS
        else:
            self.blackboard.shared_content[self.object_name] = []            
            self.agent.shared_content[self.object_name] = []
            return Status.FAILURE


# Behavior defined for GoTo Behavior
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


# Behavior defined to move towards something
class Towards(Behaviour):
    def __init__(self, name):
        super(Towards, self).__init__(name)

    def setup(self, timeout, agent):
        self.agent = agent

    def initialise(self):
        pass

    def update(self):
        return Status.SUCCESS


# Behavior defined to move away from something
class Away(Behaviour):
    def __init__(self, name):
        super(Away, self).__init__(name)

    def setup(self, timeout, agent):
        self.agent = agent

    def initialise(self):
        pass

    def update(self):
        self.agent.direction = (self.agent.direction + np.pi) % (2 * np.pi)
        return Status.SUCCESS


# Behavior defined for Randomwalk
class RandomWalk(Behaviour):
    def __init__(self, name):
        super(RandomWalk, self).__init__(name)

    def setup(self, timeout, agent):
        self.agent = agent

    def initialise(self):
        pass

    def update(self):
        delta_d = self.agent.model.random.normal(0, .1)
        self.agent.direction = (self.agent.direction + delta_d) % (2 * np.pi)
        return Status.SUCCESS


class IsMoveable(Behaviour):
    def __init__(self, name):
        super(IsMoveable, self).__init__(name)

    def setup(self, timeout, item):
        self.item = item

    def initialise(self):
        pass

    def update(self):
        try:
            if self.item.moveable:
                return Status.SUCCESS
            else:
                return Status.FAILURE
        except AttributeError:
            return Status.FAILURE


# Behavior defined to move
class Move(Behaviour):
    def __init__(self, name):
        super(Move, self).__init__(name)

    def setup(self, timeout, agent):
        self.agent = agent

    def initialise(self):
        pass

    def update(self):
        self.agent.accleration = self.agent.force / self.agent.get_weight()
        self.agent.velocity = self.agent.accleration * 1

        x = int(self.agent.location[0] + np.cos(self.agent.direction) * self.agent.speed)
        y = int(self.agent.location[1] + np.sin(self.agent.direction) * self.agent.speed)
        new_location, direction = self.agent.model.grid.check_limits((x, y), self.agent.direction)
        self.agent.model.grid.move_object(self.agent.location, self.agent, new_location)
        self.agent.location = new_location
        self.agent.direction = direction
        for item in self.agent.attached_objects:
            item.location = self.agent.location
        return Status.SUCCESS


# Behavior define for donot move
class DoNotMove(Behaviour):
    def __init__(self, name):
        super(DoNotMove, self).__init__(name)

    def setup(self, timeout, agent):
        self.agent = agent

    def initialise(self):
        pass

    def update(self):
        self.agent.moveable = False
        return Status.SUCCESS


# Behavior to check carryable attribute of an object
class IsCarryable(Behaviour):
    def __init__(self, name):
        super(IsCarryable, self).__init__(name)
        self.blackboard = Blackboard()

    def setup(self, timeout, agent, thing):
        self.agent = agent
        self.thing = thing

    def initialise(self):
        pass

    def update(self):
        objects = self.blackboard.shared_content[self.thing][0]
        try:
            if objects.carryable:
                return Status.SUCCESS
            else:
                return Status.FAILURE
        except AttributeError:
            return Status.FAILURE


# Behavior define to check is the item is carrable on its own
class IsSingleCarry(Behaviour):
    def __init__(self, name):
        super(IsSingleCarry, self).__init__(name)
        self.blackboard = Blackboard()

    def setup(self, timeout, agent, thing):
        self.agent = agent
        self.thing = thing

    def initialise(self):
        pass

    def update(self):
        # Logic to carry
        objects = self.blackboard.shared_content[self.thing][0]
        try:
            if objects.weight:
                if self.agent.capacity > objects.weight:
                    return Status.SUCCESS
            else:
                return Status.FAILURE
        except AttributeError:
            return Status.FAILURE


# Behavior define to check is the item is carrable on its own or not
class IsMultipleCarry(Behaviour):
    def __init__(self, name):
        super(IsMultipleCarry, self).__init__(name)
        self.blackboard = Blackboard()

    def setup(self, timeout, agent, thing):
        self.agent = agent
        self.thing = thing

    def initialise(self):
        pass

    def update(self):
        # Logic to carry
        objects = self.blackboard.shared_content[self.thing][0]
        try:
            if objects.weight:
                if self.agent.capacity < objects.weight:
                    return Status.SUCCESS
            else:
                return Status.FAILURE
        except AttributeError:
            return Status.FAILURE        


class SingleCarry(Behaviour):
    def __init__(self, name):
        super(SingleCarry, self).__init__(name)
        self.blackboard = Blackboard()

    def setup(self, timeout, agent, thing):
        self.agent = agent
        self.thing = thing

    def initialise(self):
        pass

    def update(self):
        # Logic to carry
        #try:
        objects = self.blackboard.shared_content[self.thing][0]
        self.agent.attached_objects.append(objects)
        self.agent.model.grid.remove_object_from_grid(objects.location, objects)
        return Status.SUCCESS
        #except:
        #    return Status.FAILURE


# class Inveter(Behaviour):
