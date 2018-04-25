from py_trees import Behaviour, Status, meta, composites, Blackboard
import numpy as np
from swarms.utils.distangle import get_direction


# Defining behaviors for the agent

class ObjectsStore:
    #def __init__(self, blackboard_content, agent_content, item):
    #    self.blackboard_content = blackboard_content
    #    self.agent_content = agent_content
    #    self.item

    @staticmethod
    def find(blackboard_content, agent_content, name):
        # Priority to blackboard
        try:
            objects = blackboard_content[name]
            return list(objects)
        except KeyError:
            try:
                objects = agent_content[name]
                return list(objects)
            except KeyError:
                return []


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
        grids = self.agent.model.grid.get_neighborhood(
            self.agent.location, self.agent.radius)
        objects = self.agent.model.grid.get_objects_from_list_of_grid(
            self.object_name, grids)
        if len(objects) >= 1:
            for item in objects:
                try:
                    name = type(item).__name__
                    self.blackboard.shared_content[name].add(item)
                    # self.agent.shared_content[name].add(item)
                except KeyError:
                    self.blackboard.shared_content[name] = {item}
                    # self.agent.shared_content[name] = {item}
                
            return Status.SUCCESS
        else:
            return Status.FAILURE


# Behavior defined for GoTo Behavior
class GoTo(Behaviour):
    def __init__(self, name):
        super(GoTo, self).__init__(name)
        self.blackboard = Blackboard()

    def setup(self, timeout, agent, item):
        self.agent = agent
        self.item = item

    def initialise(self):
        pass

    def update(self):
        try:
            objects = ObjectsStore.find(self.blackboard.shared_content, self.agent.shared_content, self.item)[0]
            self.agent.direction = get_direction(
                objects.location, self.agent.location)
            return Status.SUCCESS
        except (AttributeError, IndexError):
            return Status.FAILURE


# Behavior defined to move towards something
class Towards(Behaviour):
    def __init__(self, name):
        super(Towards, self).__init__(name)

    def setup(self, timeout, agent, thing=None):
        self.agent = agent

    def initialise(self):
        pass

    def update(self):
        return Status.SUCCESS


# Behavior defined to move away from something
class Away(Behaviour):
    def __init__(self, name):
        super(Away, self).__init__(name)

    def setup(self, timeout, agent, thing=None):
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

    def setup(self, timeout, agent, thing=None):
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
            objects = ObjectsStore.find(self.blackboard.shared_content, self.agent.shared_content, self.item)[0]
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
        pass
        """
        for item in self.agent.partial_attached_objects:
            accleration = self.agent.force / item.weight
            print(self.agent.force, item.weight)
            velocity = accleration * self.dt
            direction = self.agent.direction
            x = int(item.location[0] + np.cos(direction) * velocity)
            y = int(item.location[1] + np.sin(direction) * velocity)
            object_agent = list(item.agents.keys())[0]
            new_location, direction = object_agent.model.grid.check_limits(
                (x, y), direction)
            print('update', item.location, new_location)
            object_agent.model.grid.move_object(
                item.location, item, new_location)
            item.location = new_location
        """

    def update(self):
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
            #objects = self.blackboard.shared_content[self.thing].pop()
            objects = ObjectsStore.find(self.blackboard.shared_content, self.agent.shared_content, self.item)[0]            
            if objects.carryable:
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

    def setup(self, timeout, agent, thing):
        self.agent = agent
        self.thing = thing

    def initialise(self):
        pass

    def update(self):
        # Logic to carry
        try:
            #objects = self.blackboard.shared_content[self.thing].pop() 
            objects = ObjectsStore.find(self.blackboard.shared_content, self.agent.shared_content, self.item)[0]                       
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

    def setup(self, timeout, agent, thing):
        self.agent = agent
        self.thing = thing

    def initialise(self):
        pass

    def update(self):
        try:
            # Logic to carry
            #objects = self.blackboard.shared_content[self.thing].pop()       
            objects = ObjectsStore.find(self.blackboard.shared_content, self.agent.shared_content, self.item)[0]                 
            if objects.weight:
                if self.agent.get_capacity() < objects.weight:
                    return Status.SUCCESS
            else:
                return Status.FAILURE
        except (AttributeError, IndexError):
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
        try:
            #objects = self.blackboard.shared_content[self.thing].pop()
            objects = ObjectsStore.find(self.blackboard.shared_content, self.agent.shared_content, self.item)[0]            
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

    def setup(self, timeout, agent, thing):
        self.agent = agent
        self.thing = thing

    def initialise(self):
        pass

    def update(self):
        try:
            #objects = self.blackboard.shared_content[self.thing].pop()
            objects = ObjectsStore.find(self.blackboard.shared_content, self.agent.shared_content, self.item)[0]                        
            relative_weight = objects.calc_relative_weight()
            print('initial mc', self.agent.name, relative_weight, self.agent.get_capacity())
            if relative_weight > 0:
                if relative_weight - self.agent.get_capacity() >= 0:
                    capacity_used = self.agent.get_capacity()
                else:
                    capacity_used = relative_weight

                # Update the partial attached object
                self.agent.partial_attached_objects.append(objects)

                # Update the object so that it knows this agent has attached to it
                # self.agent.capacity_used += capacity_used
                # print('update', relative_weight, capacity_used)
                objects.agents[self.agent] = capacity_used

                return Status.SUCCESS
            else:
                # Redistribute the weights to all the attached objects
                average_weight = objects.redistribute_weights()

                self.agent.partial_attached_objects.append(objects)

                objects.agents[self.agent] = average_weight
                print('avg weig', self.agent.name, average_weight)
                return Status.SUCCESS
        except (KeyError, AttributeError, IndexError):
            return Status.FAILURE


class IsInPartialAttached(Behaviour):
    def __init__(self, name):
        super(IsInPartialAttached, self).__init__(name)
        self.blackboard = Blackboard()

    def setup(self, timeout, agent, thing):
        self.agent = agent
        self.thing = thing

    def initialise(self):
        pass

    def update(self):
        #objects = self.blackboard.shared_content[self.thing].pop()
        objects = ObjectsStore.find(self.blackboard.shared_content, self.agent.shared_content, self.item)[0]                    
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

    def setup(self, timeout, agent, thing):
        self.agent = agent
        self.thing = thing

    def initialise(self):
        pass

    def update(self):
        #objects = self.blackboard.shared_content[self.thing].pop()
        objects = ObjectsStore.find(self.blackboard.shared_content, self.agent.shared_content, self.item)[0]
        #print(self.agent.get_capacity(), objects.calc_relative_weight())
        try: 
            if self.agent.get_capacity() >= objects.calc_relative_weight():
                # self.agent.model.grid.remove_object_from_grid(
                # objects.location, objects)
                return Status.SUCCESS
            else:
                return Status.FAILURE
        except IndexError:
            return Status.FAILURE


class IsMotionTrue(Behaviour):
    def __init__(self, name):
        super(IsMotionTrue, self).__init__(name)
        # self.blackboard = Blackboard()

    def setup(self, timeout, agent, thing):
        self.agent = agent
        self.thing = thing

    def initialise(self):
        pass

    def update(self):
        # objects = self.blackboard.shared_content[self.thing][0]
        try:
            if self.agent.partial_attached_objects[0].motion is True:
                return Status.SUCCESS
            else:
                return Status.FAILURE
        except (AttributeError, IndexError):
            return Status.FAILURE


class MultipleCarry(Behaviour):
    def __init__(self, name):
        super(MultipleCarry, self).__init__(name)
        self.blackboard = Blackboard()

    def setup(self, timeout, agent, thing):
        self.agent = agent
        self.thing = thing

    def initialise(self):
        pass

    def update(self):
        #objects = self.blackboard.shared_content[self.thing].pop()
        objects = ObjectsStore.find(self.blackboard.shared_content, self.agent.shared_content, self.item)[0]
        try:
            self.agent.model.grid.remove_object_from_grid(
                objects.location, objects)
            return Status.SUCCESS
        except IndexError:
            return Status.FAILURE
        # objects = self.agent.partial_attached_objects[0]

        # Needs move function to move it
