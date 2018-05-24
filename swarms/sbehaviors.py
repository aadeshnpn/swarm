from py_trees import Behaviour, Status, Blackboard
import numpy as np
from swarms.utils.distangle import get_direction


class ObjectsStore:
    """Static class to search.

    This class provides a find method to search through
    Behavior Tree blackboard and agent content.
    """

    @staticmethod
    def find(blackboard_content, agent_content, name):
        """Let this method implement search.

        This method find implements a search through
        blackboard dictionary. If the object is not found
        in blackboard, then agent content is searched.
        """
        try:
            objects = blackboard_content[name]
            return list(objects)
        except KeyError:
            try:
                objects = agent_content[name]
                return list(objects)
            except KeyError:
                return []


class NeighbourObjects(Behaviour):
    """Sense behavior for the agents.

    Inherits the Behaviors class from py_trees. This
    behavior implements the sense function for the agents. This allows
    the agents to sense the nearby environment based on the their
    sense radius.
    """

    def __init__(self, name):
        """Init method for the sense behavior."""
        super(NeighbourObjects, self).__init__(name)
        self.blackboard = Blackboard()

    def setup(self, timeout, agent, item):
        """Have defined the setup method.

        This method defines the other objects required for the
        behavior. Agent is the actor in the environment,
        item is the name of the item we are trying to find in the
        environment and timeout defines the execution time for the
        behavior.
        """
        self.agent = agent
        self.item = item
        self.blackboard.shared_content = dict()

    def initialise(self):
        """Everytime initialization. Not required for now."""
        pass

    def update(self):
        """
        Sense the neighborhood.

        This method gets the grid values based on the current location and
        radius. The grids are used to search the environment. If the agents
        find any objects, it is stored in the behavior tree blackboard which
        is a dictionary with sets as values.
        """
        grids = self.agent.model.grid.get_neighborhood(
            self.agent.location, self.agent.radius)
        objects = self.agent.model.grid.get_objects_from_list_of_grid(
            self.item, grids)

        if len(objects) >= 1:
            for item in objects:
                try:
                    name = type(item).__name__
                    self.blackboard.shared_content[name].add(item)
                except KeyError:
                    self.blackboard.shared_content[name] = {item}
            return Status.SUCCESS
        else:
            return Status.FAILURE


class GoTo(Behaviour):
    """GoTo behavior for the agents.

    Inherits the Behaviors class from py_trees. This
    behavior implements the GoTo function for the agents. This allows
    the agents direct towards the object they want to reach. This behavior
    is only concerned with direction alignment not with movement.
    """

    def __init__(self, name):
        """Init method for the GoTo behavior."""
        super(GoTo, self).__init__(name)
        self.blackboard = Blackboard()
        self.blackboard.shared_content = dict()

    def setup(self, timeout, agent, item):
        """Have defined the setup method.

        This method defines the other objects required for the
        behavior. Agent is the actor in the environment,
        item is the name of the item we are trying to find in the
        environment and timeout defines the execution time for the
        behavior.
        """
        self.agent = agent
        self.item = item

    def initialise(self):
        """Everytime initialization. Not required for now."""
        pass

    def update(self):
        """
        Goto towards the object of interest.

        This method uses the ObjectsStore abstract class to find the
        objects sensed before and agent shared storage. If the agent
        find the object of interst in the store then, direction to the
        object of interest is computed and agent direction is set to that
        direction.
        """
        try:
            objects = ObjectsStore.find(
                self.blackboard.shared_content, self.agent.shared_content,
                self.item)[0]
            self.agent.direction = get_direction(
                objects.location, self.agent.location)
            return Status.SUCCESS
        except (AttributeError, IndexError):
            return Status.FAILURE


# Behavior defined to move towards something
class Towards(Behaviour):
    def __init__(self, name):
        super(Towards, self).__init__(name)

    def setup(self, timeout, agent, item=None):
        self.agent = agent

    def initialise(self):
        pass

    def update(self):
        return Status.SUCCESS


# Behavior defined to move away from something
class Away(Behaviour):
    def __init__(self, name):
        super(Away, self).__init__(name)

    def setup(self, timeout, agent, item=None):
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

    def setup(self, timeout, agent, item=None):
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
        self.blackboard = Blackboard()

    def setup(self, timeout, agent, item):
        self.item = item

    def initialise(self):
        pass

    def update(self):
        try:
            objects = ObjectsStore.find(
                self.blackboard.shared_content, self.agent.shared_content,
                self.item)[0]
            if objects.moveable:
                return Status.SUCCESS
            else:
                return Status.FAILURE
        except (AttributeError, IndexError):
            return Status.FAILURE


# Behavior defined to move
class Move(Behaviour):
    def __init__(self, name):
        super(Move, self).__init__(name)

    def setup(self, timeout, agent, item=None):
        self.agent = agent
        self.dt = 1.1

    def initialise(self):
        pass

    def update_partial_attached_objects(self):
        try:
            for item in self.agent.partial_attached_objects:
                accleration = self.agent.force / item.weight
                velocity = accleration * self.dt
                direction = self.agent.direction
                x = int(np.ceil(
                    item.location[0] + np.cos(direction) * velocity))
                y = int(np.ceil(
                    item.location[1] + np.sin(direction) * velocity))
                object_agent = list(item.agents.keys())[0]
                new_location, direction = object_agent.model.grid.check_limits(
                    (x, y), direction)
                object_agent.model.grid.move_object(
                    item.location, item, new_location)
                item.location = new_location
                return True
        except IndexError:
            return False

    def update(self):
        # Partially carried object
        if not self.update_partial_attached_objects():
            self.agent.accleration = self.agent.force / self.agent.get_weight()
            self.agent.velocity = self.agent.accleration * 1

            x = int(self.agent.location[0] + np.cos(
                self.agent.direction) * self.agent.velocity)
            y = int(self.agent.location[1] + np.sin(
                self.agent.direction) * self.agent.velocity)
            new_location, direction = self.agent.model.grid.check_limits(
                (x, y), self.agent.direction)
            self.agent.model.grid.move_object(
                self.agent.location, self.agent, new_location)
            self.agent.location = new_location
            self.agent.direction = direction

            # Full carried object moves along the agent
            for item in self.agent.attached_objects:
                item.location = self.agent.location

        else:
            self.agent.model.grid.move_object(
                self.agent.location, self.agent,
                self.agent.partial_attached_objects[0].location)

            self.agent.location = self.agent.partial_attached_objects[0].location

        return Status.SUCCESS


# Behavior define for donot move
class DoNotMove(Behaviour):
    def __init__(self, name):
        super(DoNotMove, self).__init__(name)

    def setup(self, timeout, agent, item=None):
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
        try:
            # objects = self.blackboard.shared_content[self.thing].pop()
            objects = ObjectsStore.find(
                self.blackboard.shared_content, self.agent.shared_content,
                self.thing)[0]
            if objects.carryable:
                return Status.SUCCESS
            else:
                return Status.FAILURE
        except (AttributeError, IndexError):
            return Status.FAILURE


# Behavior to check carryable attribute of an object
class IsDropable(Behaviour):
    def __init__(self, name):
        super(IsDropable, self).__init__(name)
        self.blackboard = Blackboard()
        self.blackboard.shared_content = dict()

    def setup(self, timeout, agent, thing):
        self.agent = agent
        self.thing = thing

    def initialise(self):
        pass

    def update(self):
        try:
            objects = ObjectsStore.find(
                self.blackboard.shared_content, self.agent.shared_content,
                self.thing)[0]
            if objects.dropable:
                return Status.SUCCESS
            else:
                return Status.FAILURE
        except (AttributeError, IndexError):
            return Status.FAILURE


# Behavior define to check is the item is carrable on its own
class IsSingleCarry(Behaviour):
    def __init__(self, name):
        super(IsSingleCarry, self).__init__(name)
        self.blackboard = Blackboard()
        self.blackboard.shared_content = dict()

    def setup(self, timeout, agent, thing):
        self.agent = agent
        self.thing = thing

    def initialise(self):
        pass

    def update(self):
        # Logic to carry
        try:
            # objects = self.blackboard.shared_content[self.thing].pop()
            objects = ObjectsStore.find(
                self.blackboard.shared_content, self.agent.shared_content,
                self.thing)[0]
            if objects.weight:
                if self.agent.get_capacity() > objects.calc_relative_weight():
                    return Status.SUCCESS
            else:
                return Status.FAILURE
        except (AttributeError, IndexError):
            return Status.FAILURE


# Behavior define to check is the item is carrable on its own or not
class IsMultipleCarry(Behaviour):
    def __init__(self, name):
        super(IsMultipleCarry, self).__init__(name)
        self.blackboard = Blackboard()
        self.blackboard.shared_content = dict()

    def setup(self, timeout, agent, thing):
        self.agent = agent
        self.thing = thing

    def initialise(self):
        pass

    def update(self):
        try:
            # Logic to carry
            # objects = self.blackboard.shared_content[self.thing].pop()
            objects = ObjectsStore.find(
                self.blackboard.shared_content, self.agent.shared_content,
                self.thing)[0]
            if objects.weight:
                if self.agent.get_capacity() < objects.weight:
                    return Status.SUCCESS
            else:
                return Status.FAILURE
        except (AttributeError, IndexError):
            return Status.FAILURE


class IsCarrying(Behaviour):
    def __init__(self, name):
        super(IsCarrying, self).__init__(name)
        self.blackboard = Blackboard()
        self.blackboard.shared_content = dict()

    def setup(self, timeout, agent, thing):
        self.agent = agent
        self.thing = thing

    def initialise(self):
        pass

    def update(self):
        try:
            objects = ObjectsStore.find(
                self.blackboard.shared_content, self.agent.shared_content,
                self.thing)[0]
            if objects in self.agent.attached_objects:
                return Status.SUCCESS
            else:
                return Status.FAILURE
        except (AttributeError, IndexError):
            return Status.FAILURE


# Behavior defined to drop the items currently carrying
class Drop(Behaviour):
    def __init__(self, name):
        super(Drop, self).__init__(name)
        self.blackboard = Blackboard()
        self.blackboard.shared_content = dict()

    def setup(self, timeout, agent, thing):
        self.agent = agent
        self.thing = thing

    def initialise(self):
        pass

    def update(self):
        try:
            objects = ObjectsStore.find(
                self.blackboard.shared_content, self.agent.shared_content,
                self.thing)[0]
            self.agent.model.grid.add_object_to_grid(objects.location, objects)
            self.agent.attached_objects.remove(objects)
            self.blackboard.shared_content['Food'].remove(objects)
            return Status.SUCCESS
        except (AttributeError, IndexError):
            return Status.FAILURE


class DropPartial(Behaviour):
    def __init__(self, name):
        super(DropPartial, self).__init__(name)
        self.blackboard = Blackboard()
        self.blackboard.shared_content = dict()

    def setup(self, timeout, agent, thing):
        self.agent = agent
        self.thing = thing

    def initialise(self):
        pass

    def update(self):
        try:
            objects = self.agent.partial_attached_objects[0]
            objects.agents.pop(self.agent)
            self.agent.model.grid.add_object_to_grid(objects.location, objects)
            self.agent.partial_attached_objects.remove(objects)
            return Status.SUCCESS

        except (AttributeError, IndexError):
            return Status.FAILURE


# Behavior defined to carry the items found
class SingleCarry(Behaviour):
    def __init__(self, name):
        super(SingleCarry, self).__init__(name)
        self.blackboard = Blackboard()
        self.blackboard.shared_content = dict()

    def setup(self, timeout, agent, thing):
        self.agent = agent
        self.thing = thing

    def initialise(self):
        pass

    def update(self):
        try:
            # objects = self.blackboard.shared_content[self.thing].pop()
            objects = ObjectsStore.find(
                self.blackboard.shared_content, self.agent.shared_content,
                self.thing)[0]
            self.agent.attached_objects.append(objects)
            self.agent.model.grid.remove_object_from_grid(
                objects.location, objects)
            return Status.SUCCESS
        except (AttributeError, IndexError):
            return Status.FAILURE


class InitiateMultipleCarry(Behaviour):
    def __init__(self, name):
        super(InitiateMultipleCarry, self).__init__(name)
        self.blackboard = Blackboard()
        self.blackboard.shared_content = dict()

    def setup(self, timeout, agent, item):
        self.agent = agent
        self.item = item

    def initialise(self):
        pass

    def update(self):
        try:
            # objects = self.blackboard.shared_content[self.thing].pop()
            objects = ObjectsStore.find(
                self.blackboard.shared_content, self.agent.shared_content,
                self.item)[0]
            relative_weight = objects.calc_relative_weight()
            if relative_weight > 0:
                if relative_weight - self.agent.get_capacity() >= 0:
                    capacity_used = self.agent.get_capacity()
                else:
                    capacity_used = relative_weight

                # Update the partial attached object
                self.agent.partial_attached_objects.append(objects)

                # Update the object so that it knows this agent
                # has attached to it
                objects.agents[self.agent] = capacity_used

                return Status.SUCCESS
            else:
                # Redistribute the weights to all the attached objects
                average_weight = objects.redistribute_weights()

                self.agent.partial_attached_objects.append(objects)

                objects.agents[self.agent] = average_weight

                return Status.SUCCESS
        except (KeyError, AttributeError, IndexError):
            return Status.FAILURE


class IsInPartialAttached(Behaviour):
    def __init__(self, name):
        super(IsInPartialAttached, self).__init__(name)
        self.blackboard = Blackboard()
        self.blackboard.shared_content = dict()

    def setup(self, timeout, agent, item):
        self.agent = agent
        self.item = item

    def initialise(self):
        pass

    def update(self):
        # objects = self.blackboard.shared_content[self.thing].pop()
        objects = ObjectsStore.find(
            self.blackboard.shared_content, self.agent.shared_content,
            self.item)[0]
        # print (self.agent, objects,
        #  self.agent.partial_attached_objects, objects.agents)
        try:
            if objects in self.agent.partial_attached_objects and \
                    self.agent in objects.agents:
                return Status.SUCCESS
            else:
                return Status.FAILURE
        except IndexError:
            return Status.FAILURE


class IsEnoughStrengthToCarry(Behaviour):
    def __init__(self, name):
        super(IsEnoughStrengthToCarry, self).__init__(name)
        self.blackboard = Blackboard()
        self.blackboard.shared_content = dict()

    def setup(self, timeout, agent, item):
        self.agent = agent
        self.item = item

    def initialise(self):
        pass

    def update(self):
        objects = ObjectsStore.find(
            self.blackboard.shared_content, self.agent.shared_content,
            self.item)[0]
        try:
            if self.agent.get_capacity() >= objects.calc_relative_weight():
                return Status.SUCCESS
            else:
                return Status.FAILURE
        except IndexError:
            return Status.FAILURE


class IsMotionTrue(Behaviour):
    def __init__(self, name):
        super(IsMotionTrue, self).__init__(name)
        self.blackboard = Blackboard()
        self.blackboard.shared_content = dict()

    def setup(self, timeout, agent, item):
        self.agent = agent
        self.item = item

    def initialise(self):
        pass

    def update(self):
        try:
            if self.agent.partial_attached_objects[0].motion is True:
                return Status.SUCCESS
            else:
                return Status.FAILURE
        except (AttributeError, IndexError):
            return Status.FAILURE


class IsVisitedBefore(Behaviour):
    def __init__(self, name):
        super(IsVisitedBefore, self).__init__(name)
        self.blackboard = Blackboard()

    def setup(self, timeout, agent, item):
        self.agent = agent
        self.item = item

    def initialise(self):
        pass

    def update(self):
        try:
            objects = ObjectsStore.find(
                self.blackboard.shared_content, self.agent.shared_content,
                self.item)[0]
            if objects:
                return Status.SUCCESS
            else:
                return Status.FAILURE
        except (AttributeError, IndexError):
            return Status.FAILURE


class MultipleCarry(Behaviour):
    def __init__(self, name):
        super(MultipleCarry, self).__init__(name)
        self.blackboard = Blackboard()
        self.blackboard.shared_content = dict()

    def setup(self, timeout, agent, item):
        self.agent = agent
        self.item = item

    def initialise(self):
        pass

    def update(self):
        objects = ObjectsStore.find(
            self.blackboard.shared_content, self.agent.shared_content,
            self.item)[0]
        try:
            self.agent.model.grid.remove_object_from_grid(
                objects.location, objects)
            return Status.SUCCESS
        except IndexError:
            return Status.FAILURE
