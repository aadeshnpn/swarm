"""Defines all the primitive behaviors for the agents.

This file name is sbehaviors coz `s` stands for swarms.
"""

from py_trees import Behaviour, Status, Blackboard
import numpy as np
from swarms.utils.distangle import get_direction
from swarms.lib.objects import Signal, Cue


class ObjectsStore:
    """Static class to search.

    This class provides a find method to search through
    Behavior Tree blackboard and agent content.
    """

    @staticmethod
    def find(blackboard_content, agent_content, name, agent_name):
        """Let this method implement search.

        This method find implements a search through
        blackboard dictionary. If the object is not found
        in blackboard, then agent content is searched.
        """
        try:
            objects = blackboard_content[name + str(agent_name)]
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

    def receive_signals(self):
        """Receive signals from other agents.

        Since this is the primary behavior for the agents to sense
        the environment, we include the receive signal method here.
        The agents will be able to
        sense the environment and check if
        it receives any signals from other agents.
        """

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
        # Need to reset blackboard contents after each sense
        self.blackboard.shared_content = dict()

        if len(objects) >= 1:
            for item in objects:
                name = type(item).__name__
                # Is the item is not carrable, its location
                # and property doesnot change. So we can commit its
                # information to memory
                if item.carryable is False:
                    try:
                        self.agent.shared_content[name].add(item)
                    except KeyError:
                        self.agent.shared_content[name] = {item}
                else:
                    name = name + str(self.agent.name)
                    try:
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
                self.item, self.agent.name)[0]
            self.agent.direction = get_direction(
                objects.location, self.agent.location)
            return Status.SUCCESS
        except (AttributeError, IndexError):
            return Status.FAILURE


# Behavior defined to move towards something
class Towards(Behaviour):
    """Towards behaviors.

    Changes the direction to go towards the object.
    """

    def __init__(self, name):
        """Initialize."""
        super(Towards, self).__init__(name)

    def setup(self, timeout, agent, item=None):
        """Setup."""
        self.agent = agent

    def initialise(self):
        """Pass."""
        pass

    def update(self):
        """Nothing much to do."""
        return Status.SUCCESS


# Behavior defined to move away from something
class Away(Behaviour):
    """Away behavior."""

    def __init__(self, name):
        """Initialize."""
        super(Away, self).__init__(name)

    def setup(self, timeout, agent, item=None):
        """Setup."""
        self.agent = agent

    def initialise(self):
        """Pass."""
        pass

    def update(self):
        """Compute direction and negate it."""
        self.agent.direction = (self.agent.direction + np.pi) % (2 * np.pi)
        return Status.SUCCESS


# Behavior defined for Randomwalk
class RandomWalk(Behaviour):
    """Random walk behaviors."""

    def __init__(self, name):
        """Initialize."""
        super(RandomWalk, self).__init__(name)

    def setup(self, timeout, agent, item=None):
        """Setup."""
        self.agent = agent

    def initialise(self):
        """Pass."""
        pass

    def update(self):
        """Compute random direction and set it to agent direction."""
        delta_d = self.agent.model.random.normal(0, .1)
        self.agent.direction = (self.agent.direction + delta_d) % (2 * np.pi)
        return Status.SUCCESS


class IsMoveable(Behaviour):
    """Check is the item is moveable."""

    def __init__(self, name):
        """Initialize."""
        super(IsMoveable, self).__init__(name)
        self.blackboard = Blackboard()

    def setup(self, timeout, agent, item):
        """Setup."""
        self.agent = agent
        self.item = item

    def initialise(self):
        """Pass."""
        pass

    def update(self):
        """Get the object and check its movelable property."""
        try:
            objects = ObjectsStore.find(
                self.blackboard.shared_content, self.agent.shared_content,
                self.item, self.agent.name)[0]
            if objects.moveable:
                return Status.SUCCESS
            else:
                return Status.FAILURE
        except (AttributeError, IndexError):
            return Status.FAILURE


# Behavior defined to move
class Move(Behaviour):
    """Actually move the agent.

    Move the agent with any other object fully attached or
    partially attached to the agent.
    """

    def __init__(self, name):
        """Initialize."""
        super(Move, self).__init__(name)

    def setup(self, timeout, agent, item=None):
        """Setup."""
        self.agent = agent
        self.dt = 1.1

    def initialise(self):
        """Pass."""
        pass

    def update_signals(self, old_loc, new_loc):
        """Signal also move along with agents.

        Signal is created by the agent. It has certain broadcast radius. It
        moves along with the agent. So this move behavior should also be
        responsible to move the signals.
        """
        try:
            for signal in self.agent.signals:
                self.agent.model.grid.move_object(
                    old_loc, signal, new_loc)
        except IndexError:
            pass

    def update_partial_attached_objects(self):
        """Move logic for partially attached objects."""
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
        """Move logic for agent and fully carried object."""
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

            # Now the agent location has been updated, update the signal grids
            self.update_signals(self.agent.location, new_location)

            self.agent.location = new_location
            self.agent.direction = direction

            # Full carried object moves along the agent
            for item in self.agent.attached_objects:
                item.location = self.agent.location

        else:
            new_location = self.agent.partial_attached_objects[0].location
            self.agent.model.grid.move_object(
                self.agent.location, self.agent, new_location)

            # Now the agent location has been updated, update the signal grids
            self.update_signals(self.agent.location, new_location)

            self.agent.location = new_location

        return Status.SUCCESS


# Behavior define for donot move
class DoNotMove(Behaviour):
    """Stand still behaviors."""

    def __init__(self, name):
        """Initialize."""
        super(DoNotMove, self).__init__(name)

    def setup(self, timeout, agent, item=None):
        """Setup."""
        self.agent = agent

    def initialise(self):
        """Pass."""
        pass

    def update(self):
        """Update agent moveable property."""
        self.agent.moveable = False
        return Status.SUCCESS


# Behavior to check carryable attribute of an object
class IsCarryable(Behaviour):
    """Check carryable attribute of the item."""

    def __init__(self, name):
        """Initialize."""
        super(IsCarryable, self).__init__(name)
        self.blackboard = Blackboard()
        self.blackboard.shared_content = dict()

    def setup(self, timeout, agent, thing):
        """Setup."""
        self.agent = agent
        self.thing = thing

    def initialise(self):
        """Pass."""
        pass

    def update(self):
        """Check carryable property."""
        try:
            objects = ObjectsStore.find(
                self.blackboard.shared_content, self.agent.shared_content,
                self.thing, self.agent.name)[0]
            if objects.carryable:
                return Status.SUCCESS
            else:
                return Status.FAILURE
        except (AttributeError, IndexError):
            return Status.FAILURE


# Behavior to check carryable attribute of an object
class IsDropable(Behaviour):
    """Check dropable property."""

    def __init__(self, name):
        """Initialize."""
        super(IsDropable, self).__init__(name)
        self.blackboard = Blackboard()
        self.blackboard.shared_content = dict()

    def setup(self, timeout, agent, thing):
        """Setup."""
        self.agent = agent
        self.thing = thing

    def initialise(self):
        """Pass."""
        pass

    def update(self):
        """Check the dropable attribute."""
        try:
            objects = ObjectsStore.find(
                self.blackboard.shared_content, self.agent.shared_content,
                self.thing, self.agent.name)[0]
            if objects.dropable:
                return Status.SUCCESS
            else:
                return Status.FAILURE
        except (AttributeError, IndexError):
            return Status.FAILURE


# Behavior define to check is the item is carrable on its own
class IsSingleCarry(Behaviour):
    """Single carry behavior."""

    def __init__(self, name):
        """Initialize."""
        super(IsSingleCarry, self).__init__(name)
        self.blackboard = Blackboard()
        self.blackboard.shared_content = dict()

    def setup(self, timeout, agent, thing):
        """Setup."""
        self.agent = agent
        self.thing = thing

    def initialise(self):
        """Pass."""
        pass

    def update(self):
        """Logic to check if the object can be carried by single agent."""
        # Logic to carry
        try:
            objects = ObjectsStore.find(
                self.blackboard.shared_content, self.agent.shared_content,
                self.thing, self.agent.name)[0]
            if objects.weight:
                if self.agent.get_capacity() > objects.calc_relative_weight():
                    return Status.SUCCESS
            else:
                return Status.FAILURE
        except (AttributeError, IndexError):
            return Status.FAILURE


# Behavior define to check is the item is carrable on its own or not
class IsMultipleCarry(Behaviour):
    """Multiple carry behaviour."""

    def __init__(self, name):
        """Initialize."""
        super(IsMultipleCarry, self).__init__(name)
        self.blackboard = Blackboard()
        self.blackboard.shared_content = dict()

    def setup(self, timeout, agent, thing):
        """Setup."""
        self.agent = agent
        self.thing = thing

    def initialise(self):
        """Pass."""
        pass

    def update(self):
        """Logic for multiple carry by checking the weights."""
        try:
            # Logic to carry
            # objects = self.blackboard.shared_content[self.thing].pop()
            objects = ObjectsStore.find(
                self.blackboard.shared_content, self.agent.shared_content,
                self.thing, self.agent.name)[0]
            if objects.weight:
                if self.agent.get_capacity() < objects.weight:
                    return Status.SUCCESS
            else:
                return Status.FAILURE
        except (AttributeError, IndexError):
            return Status.FAILURE


class IsCarrying(Behaviour):
    """Condition check if the agent is carrying something."""

    def __init__(self, name):
        """Initialize."""
        super(IsCarrying, self).__init__(name)
        self.blackboard = Blackboard()
        self.blackboard.shared_content = dict()

    def setup(self, timeout, agent, item):
        """Setup."""
        self.agent = agent
        self.item = item

    def initialise(self):
        """Pass."""
        pass

    def update(self):
        """Logic for object carrying check."""
        try:
            things = []
            for item in self.agent.attached_objects:
                things.append(type(item).__name__)

            if self.item in set(things):
                return Status.SUCCESS
            else:
                return Status.FAILURE
        except (AttributeError, IndexError):
            return Status.FAILURE


# Behavior defined to drop the items currently carrying
class Drop(Behaviour):
    """Drop behavior to drop items which is being carried."""

    def __init__(self, name):
        """Initialize."""
        super(Drop, self).__init__(name)
        self.blackboard = Blackboard()
        self.blackboard.shared_content = dict()

    def setup(self, timeout, agent, item):
        """Setup."""
        self.agent = agent
        self.item = item

    def initialise(self):
        """Pass."""
        pass

    def update(self):
        """Logic to drop the item."""
        try:
            # Get the objects from the actuators
            objects = list(filter(
                lambda x: type(x).__name__ == self.item,
                self.agent.attached_objects))[0]

            self.agent.model.grid.add_object_to_grid(objects.location, objects)
            self.agent.attached_objects.remove(objects)
            objects.agent_name = self.agent.name

            # Temporary fix
            # Store the genome which activated the single carry
            objects.phenotype['drop'] = self.agent.individual[0].phenotype

            # objects.agents.remove(self.agent)
            return Status.SUCCESS
        except (AttributeError, IndexError):
            return Status.FAILURE


class DropPartial(Behaviour):
    """Drop behavior for partially attached object."""

    def __init__(self, name):
        """Initialize."""
        super(DropPartial, self).__init__(name)
        self.blackboard = Blackboard()
        self.blackboard.shared_content = dict()

    def setup(self, timeout, agent, item):
        """Setup."""
        self.agent = agent
        self.item = item

    def initialise(self):
        """Pass."""
        pass

    def update(self):
        """Logic to drop partially attached object."""
        try:
            objects = list(filter(
                lambda x: type(x).__name__ == self.item,
                self.agent.partial_attached_objects))[0]

            # objects = self.agent.partial_attached_objects[0]
            objects.agents.pop(self.agent)
            self.agent.model.grid.add_object_to_grid(objects.location, objects)
            self.agent.partial_attached_objects.remove(objects)
            return Status.SUCCESS

        except (AttributeError, IndexError):
            return Status.FAILURE


# Behavior defined to carry the items found
class SingleCarry(Behaviour):
    """Carry behavior."""

    def __init__(self, name):
        """Initialize."""
        super(SingleCarry, self).__init__(name)
        self.blackboard = Blackboard()
        self.blackboard.shared_content = dict()

    def setup(self, timeout, agent, item):
        """Setup."""
        self.agent = agent
        self.item = item

    def initialise(self):
        """Pass."""
        pass

    def update(self):
        """Carry logic to carry the object by the agent."""
        try:
            objects = ObjectsStore.find(
                self.blackboard.shared_content, self.agent.shared_content,
                self.item, self.agent.name)[0]
            self.agent.attached_objects.append(objects)
            self.agent.model.grid.remove_object_from_grid(
                objects.location, objects)
            objects.agent_name = self.agent.name
            # Add the agent to the object dict
            # objects.agents[self.agent] = self.agent.get_capacity()

            # Temporary fix
            # Store the genome which activated the single carry
            objects.phenotype = {'carry': self.agent.individual[0].phenotype}
            return Status.SUCCESS
        except (AttributeError, IndexError):
            return Status.FAILURE
        except ValueError:
            self.agent.attached_objects.remove(objects)
            return Status.FAILURE


class InitiateMultipleCarry(Behaviour):
    """Behavior to initiate multiple carry process."""

    def __init__(self, name):
        """Initialize."""
        super(InitiateMultipleCarry, self).__init__(name)
        self.blackboard = Blackboard()
        self.blackboard.shared_content = dict()

    def setup(self, timeout, agent, item):
        """Setup."""
        self.agent = agent
        self.item = item

    def initialise(self):
        """Pass."""
        pass

    def update(self):
        """Logic to initiaite multiple carry process."""
        try:
            # objects = self.blackboard.shared_content[self.thing].pop()
            objects = ObjectsStore.find(
                self.blackboard.shared_content, self.agent.shared_content,
                self.item, self.agent.name)[0]
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
    """Condition to check if the object is in partially attached list."""

    def __init__(self, name):
        """Initialize."""
        super(IsInPartialAttached, self).__init__(name)
        self.blackboard = Blackboard()
        self.blackboard.shared_content = dict()

    def setup(self, timeout, agent, item):
        """Setup."""
        self.agent = agent
        self.item = item

    def initialise(self):
        """Pass."""
        pass

    def update(self):
        """Logic to check if the object is in partially attached list."""
        # objects = self.blackboard.shared_content[self.thing].pop()
        try:
            things = []

            for item in self.agent.partial_attached_objects:
                things.append(type(item).__name__)

            objects = list(filter(
                lambda x: type(x).__name__ == self.item,
                self.agent.partial_attached_objects))[0]

            if self.item in set(things) and \
                    self.agent in objects.agents:
                return Status.SUCCESS
            else:
                return Status.FAILURE
        except IndexError:
            return Status.FAILURE


class IsEnoughStrengthToCarry(Behaviour):
    """Condition to check if the agent has enough strength to carry."""

    def __init__(self, name):
        """Initialize."""
        super(IsEnoughStrengthToCarry, self).__init__(name)
        self.blackboard = Blackboard()
        self.blackboard.shared_content = dict()

    def setup(self, timeout, agent, item):
        """Setup."""
        self.agent = agent
        self.item = item

    def initialise(self):
        """Pass."""
        pass

    def update(self):
        """Logic to check if the agent has enough strength to carry."""
        try:
            objects = ObjectsStore.find(
                self.blackboard.shared_content, self.agent.shared_content,
                self.item, self.agent.name)[0]

            if self.agent.get_capacity() >= objects.calc_relative_weight():
                return Status.SUCCESS
            else:
                return Status.FAILURE
        except IndexError:
            return Status.FAILURE


class IsMotionTrue(Behaviour):
    """Condition to check is the object is moving."""

    def __init__(self, name):
        """Initialize."""
        super(IsMotionTrue, self).__init__(name)
        self.blackboard = Blackboard()
        self.blackboard.shared_content = dict()

    def setup(self, timeout, agent, item):
        """Setup."""
        self.agent = agent
        self.item = item

    def initialise(self):
        """Pass."""
        pass

    def update(self):
        """Logic to check if the object is moving."""
        try:
            if self.agent.partial_attached_objects[0].motion is True:
                return Status.SUCCESS
            else:
                return Status.FAILURE
        except (AttributeError, IndexError):
            return Status.FAILURE


class IsVisitedBefore(Behaviour):
    """Condition to check is the object is visited before."""

    def __init__(self, name):
        """Initialize."""
        super(IsVisitedBefore, self).__init__(name)
        self.blackboard = Blackboard()

    def setup(self, timeout, agent, item):
        """Setup."""
        self.agent = agent
        self.item = item

    def initialise(self):
        """Pass."""
        pass

    def update(self):
        """Logic to check is the object is visited before."""
        try:
            objects = ObjectsStore.find(
                self.blackboard.shared_content, self.agent.shared_content,
                self.item, self.agent.name)[0]
            if objects:
                return Status.SUCCESS
            else:
                return Status.FAILURE
        except (AttributeError, IndexError):
            return Status.FAILURE


class MultipleCarry(Behaviour):
    """Multiple carry behavior."""

    def __init__(self, name):
        """Initialize."""
        super(MultipleCarry, self).__init__(name)
        self.blackboard = Blackboard()
        self.blackboard.shared_content = dict()

    def setup(self, timeout, agent, item):
        """Setup."""
        self.agent = agent
        self.item = item

    def initialise(self):
        """Pass."""
        pass

    def update(self):
        """Logic for multiple carry."""
        try:
            objects = ObjectsStore.find(
                self.blackboard.shared_content, self.agent.shared_content,
                self.item, self.agent.name)[0]

            self.agent.model.grid.remove_object_from_grid(
                objects.location, objects)
            return Status.SUCCESS
        except IndexError:
            return Status.FAILURE


# Lets start some communication behaviors
class SignalDoesNotExists(Behaviour):
    """Signal exists behavior.

    This behavior enables agents to check it that signal already exists.
    """

    def __init__(self, name):
        """Initialize."""
        super(SignalDoesNotExists, self).__init__(name)
        self.blackboard = Blackboard()
        self.blackboard.shared_content = dict()

    def setup(self, timeout, agent, item):
        """Setup."""
        self.agent = agent
        self.item = item

    def initialise(self):
        """Pass."""
        pass

    def update(self):
        """Logic for sending signal."""
        try:
            # Find the object the agent is trying to signal
            objects = ObjectsStore.find(
                self.blackboard.shared_content, self.agent.shared_content,
                self.item, self.agent.name)[0]

            if len(self.agent.signals) > 0:
                # Check the agetns signals array for its exitance
                signal_objects = []
                for signal in self.agent.signals:
                    signal_objects.append(signal.object_to_communicate)

                if objects not in signal_objects:
                    return Status.SUCCESS
                else:
                    return Status.FAILURE
            else:
                return Status.SUCCESS

        except (IndexError, AttributeError):
            return Status.FAILURE


class SendSignal(Behaviour):
    """Signalling behavior.

    This behavior enables agents to send signals about the information they
    have gathered. The information could be about location of site, hub, food,
    obstacles and others.
    """

    def __init__(self, name):
        """Initialize."""
        super(SendSignal, self).__init__(name)
        self.blackboard = Blackboard()
        self.blackboard.shared_content = dict()

    def setup(self, timeout, agent, item):
        """Setup."""
        self.agent = agent
        self.item = item

    def initialise(self):
        """Pass."""
        pass

    def update(self):
        """Logic for sending signal."""
        try:
            objects = ObjectsStore.find(
                self.blackboard.shared_content, self.agent.shared_content,
                self.item, self.agent.name)[0]

            # Initialize the signal object
            signal = Signal(
                id=self.agent.name, location=self.agent.location,
                radius=self.agent.radius, object_to_communicate=objects)

            # Add the signal to the grids so it could be sensed by
            # other agents
            self.agent.model.grid.add_object_to_grid(
                self.agent.location, signal)

            # Append the signal object to the agent signal list
            self.agent.signals.append(signal)

            return Status.SUCCESS
        except (IndexError, AttributeError):
            return Status.FAILURE


class ReceiveSignal(Behaviour):
    """Receive signals from other agents.

    Since this is the primary behavior for the agents to sense
    the environment, we include the receive signal method here.
    The agents will be able to sense the environment and check if
    it receives any signals from other agents.
    """

    def __init__(self, name):
        """Initialize."""
        super(ReceiveSignal, self).__init__(name)
        self.blackboard = Blackboard()
        self.blackboard.shared_content = dict()

    def setup(self, timeout, agent, item='Signal'):
        """Setup."""
        self.agent = agent
        self.item = item

    def initialise(self):
        """Pass."""
        pass

    def update(self):
        """Logic for receiving signal."""
        try:
            objects = ObjectsStore.find(
                self.blackboard.shared_content, self.agent.shared_content,
                self.item, self.agent.name)[0]
            # Extract the information from the signal object and
            # store into the agent memory
            objects = objects.communicated_object
            name = type(objects).__name__
            try:
                self.agent.shared_content[name].add(objects)
            except KeyError:
                self.agent.shared_content[name] = {objects}
            return Status.SUCCESS
        except (IndexError, AttributeError):
            return Status.FAILURE


class CueDoesNotExists(Behaviour):
    """Cue does not exists behavior.

    This behavior enables agents to check if that cue already exists.
    """

    def __init__(self, name):
        """Initialize."""
        super(CueDoesNotExists, self).__init__(name)
        self.blackboard = Blackboard()
        self.blackboard.shared_content = dict()

    def setup(self, timeout, agent, item):
        """Setup."""
        self.agent = agent
        self.item = item

    def initialise(self):
        """Pass."""
        pass

    def update(self):
        """Logic for cue checking."""
        try:
            # Find the object the agent is trying to cue
            grids = self.agent.model.grid.get_neighborhood(
                self.agent.location, self.agent.radius)
            cue_objects = self.agent.model.grid.get_objects_from_list_of_grid(
                'Cue', grids)

            if len(cue_objects) > 0:
                # Check the agetns cue list for its exitance
                objects = ObjectsStore.find(
                    self.blackboard.shared_content, self.agent.shared_content,
                    self.item, self.agent.name)[0]
                cue_in_list = [
                    cue.object_to_communicate for cue in cue_objects]
                if objects not in cue_in_list:
                    return Status.SUCCESS
                else:
                    return Status.FAILURE
            else:
                return Status.SUCCESS

        except (IndexError, AttributeError):
            return Status.FAILURE


# Communication behaviors related to cue
class DropCue(Behaviour):
    """Drop cue in the environment.

    This is a communication behavior where a physical object
    is placed in the environment which gives a particular information
    to the agents sensing this cue.
    """

    def __init__(self, name):
        """Initialize."""
        super(DropCue, self).__init__(name)
        self.blackboard = Blackboard()
        self.blackboard.shared_content = dict()

    def setup(self, timeout, agent, item):
        """Setup."""
        self.agent = agent
        self.item = item

    def initialise(self):
        """Pass."""
        pass

    def update(self):
        """Logic for dropping cue."""
        try:
            objects = ObjectsStore.find(
                self.blackboard.shared_content, self.agent.shared_content,
                self.item, self.agent.name)[0]

            # Initialize the cue object
            cue = Cue(
                id=self.agent.name, location=self.agent.location,
                radius=self.agent.radius, object_to_communicate=objects)

            # Add the cue to the grids so it could be sensed by
            # other agents
            self.agent.model.grid.add_object_to_grid(
                cue.location, cue)

            # We just drop the cue on the environment and don't keep track
            # of it. Instead of using cue here we can derive a class from cue
            # and call it pheromonone
            return Status.SUCCESS
        except (IndexError, AttributeError):
            return Status.FAILURE


class PickCue(Behaviour):
    """Pick cue in the environment.

    This is a communication behavior where the information from the cue
    object in the environment is pickedup.
    """

    def __init__(self, name):
        """Initialize."""
        super(PickCue, self).__init__(name)
        self.blackboard = Blackboard()
        self.blackboard.shared_content = dict()

    def setup(self, timeout, agent, item='Cue'):
        """Setup."""
        self.agent = agent
        self.item = item

    def initialise(self):
        """Pass."""
        pass

    def update(self):
        """Logic for pickup cue."""
        try:
            objects = ObjectsStore.find(
                self.blackboard.shared_content, self.agent.shared_content,
                self.item, self.agent.name)[0]

            # Get information from the cue. For now, the agents orients
            # its direction towards the object that is communicated
            self.agent.direction = get_direction(
                objects.communicated_location, self.agent.location)
            objects = objects.communicated_object
            name = type(objects).__name__
            try:
                self.agent.shared_content[name].add(objects)
            except KeyError:
                self.agent.shared_content[name] = {objects}

            return Status.SUCCESS
        except (IndexError, AttributeError):
            return Status.FAILURE
