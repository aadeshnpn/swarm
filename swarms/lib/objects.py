"""Base class for all the objects that can be defined in the environment."""

import numpy as np


class EnvironmentObject:
    """Base environment object."""

    def __init__(self, id=1, location=(0, 0), radius=20):
        """Initialize."""
        self.id = id
        self.location = location
        self.radius = radius
        self.dropable = False
        self.carryable = False
        self.passable = False
        self.deathable = False
        self.moveable = False

# Class to define hub object
class Hub(EnvironmentObject):
    """Hub object."""

    def __init__(self, id=1, location=(0, 0), radius=20):
        """Initialize."""
        super().__init__(id, location, radius)
        self.dropable = True
        self.carryable = False
        self.passable = True
        self.deathable = False
        self.moveable = False


# Class to define site object
class Sites(EnvironmentObject):
    """Site object."""

    def __init__(self, id=1, location=(0, 0), radius=20, q_value=0.5):
        """Initialize.

        Sites will be the agents new hub
        """
        super().__init__(id, location, radius)
        self.q_value = q_value
        self.dropable = False
        self.carryable = False
        self.passable = True
        self.deathable = False
        self.moveable = False


# Class to define source object
class Source(EnvironmentObject):
    """Source object."""

    def __init__(self, id=1, location=(0, 0), radius=20, food_units=0.5):
        """Initialize.

        Source contains food unit but is not carryable.
        """
        super().__init__(id, location, radius)
        # self.food_units = self.q_value * 1000
        self.food_units = food_units
        self.dropable = False
        self.carryable = False
        self.passable = True
        self.deathable = False
        self.moveable = False

# Class to define obstacle
class Obstacles(EnvironmentObject):
    """Obstacle object."""

    def __init__(self, id=1, location=(0, 0), radius=20):
        """Initialize."""
        super().__init__(id, location, radius)
        self.potential_field = None
        self.dropable = True
        self.carryable = False
        self.passable = False
        self.deathable = False
        self.moveable = False


# Class to define carryable property
class Carryable(EnvironmentObject):
    """Carryable class of object."""

    def __init__(self, id=1, location=(0, 0), radius=20):
        """Initialize.

        Carrable object can be carried by agents. So attributes are
        added to make that happen.
        """
        super().__init__(id, location, radius)
        # Carryable boolen value
        self.carryable = True
        self.dropable = False
        self.passable = True
        self.deathable = False
        self.moveable = False

        self.weight = int(self.radius / 2)
        self.capacity = self.weight
        self.motion = False
        self.agents = dict()
        self.direction = 0

    def calc_relative_weight(self):
        """Compute relative weight of the object."""
        relative_weight = self.weight
        for agent in self.agents:
            if relative_weight > 0:
                relative_weight -= self.agents[agent]
        return relative_weight

    def normalize_weights_on_agents(self):
        """Normamlize the weights."""
        average_weight = self.weight
        if len(self.agents) > 1:
            average_weight = self.weight / len(self.agents)

        """
        if average_weight <= 1:
            return 0
        else:
            weight_remains = 0
            self.agents.sort(key=lambda x: x.used_capacity, reverse=True)
            for agent in self.agents:
                if agent.capacity >= average_weight:
                    agent.used_capacity = average_weight
                else
        """
        return average_weight

    def redistribute_weights(self):
        """Redistribute the weight."""
        average_weight = self.weight / (len(self.agents) + 1)
        for agent in self.agents:
            self.agents[agent] = average_weight
        return average_weight

    def calc_totalforces(self):
        """Compute total force."""
        total_force = 0
        for agent in self.agents:
            total_force += agent.force
        return total_force

    def calc_direction(self):
        """Compute direction."""
        average_direction = 0
        for agent in self.agents:
            average_direction += agent.direction
        return average_direction % (2 * np.pi)


# Class to define communication
class Communication(EnvironmentObject):
    """Base class for communication."""

    def __init__(self, id=1, location=(
            0, 0), radius=20, object_to_communicate=None):
        """Initialize."""
        super().__init__(id, location, radius)
        # Communication parameters for signal
        self.communicated_object = object_to_communicate
        self.communicated_location = self.communicated_object.location

        self.dropable = False
        self.carryable = False
        self.passable = False
        self.deathable = False
        self.moveable = False


# Class to define signal
class Signal(Communication):
    """Signal object which broadcasts information."""

    def __init__(self, id=1, location=(
            0, 0), radius=20, object_to_communicate=None):
        """Initialize."""
        super().__init__(id, None, radius, object_to_communicate)
        self.dropable = False
        self.carryable = False
        self.passable = True
        self.deathable = False
        self.moveable = False


# Class to define Cue
class Cue(Communication):
    """Cue object with provides stationary information."""

    def __init__(self, id=1, location=(
            0, 0), radius=20, object_to_communicate=None):
        """Initialize."""
        super().__init__(id, location, radius, object_to_communicate)
        self.dropable = False
        self.carryable = False
        self.passable = True
        self.deathable = False
        self.moveable = False


# Class to define Traps
class Traps(EnvironmentObject):
    """Trap object which kills the agents."""

    def __init__(self, id=1, location=(0, 0), radius=20):
        """Initialize."""
        super().__init__(id, location, radius)
        self.dropable = False
        self.carryable = False
        self.passable = False
        self.deathable = True
        self.moveable = False


# Class to define Food
class Food(Carryable):
    """Food object which is carried by agents."""

    def __init__(self, id=1, location=(0, 0), radius=2):
        """Initialize."""
        super().__init__(id, location, radius)


# Class to define Derbis
class Debris(Carryable):
    """Debris object."""

    def __init__(self, id=1, location=(0, 0), radius=2, weight=5):
        """Initialize."""
        super().__init__(id, location, radius)


# Class to define pheromones
class Pheromones(EnvironmentObject):
    """Base class for Pheromones."""
    def __init__(
            self, id=1, location=(0, 0), radius=3, expire_time=20,
            attractive=True, direction=0.0):
        """Initialize."""
        super().__init__(id, location, radius)
        self.dropable = True
        self.carryable = False
        self.passable = True
        self.deathable = False
        self.moveable = False
        self.expire_time = expire_time
        self.attractive = attractive
        self.direction = direction
        self.strength = np.round(np.exp(-1*np.array(list(range(-2, self.expire_time)))))
        self.current_time = 0

    def step(self):
        self.current_time += 1
