# Base class for all the objects that can be defined in the environment
class EnvironmentObject:
    def __init__(self, id=1, location=(0, 0), radius=20):
        self.id = id
        self.location = location
        self.radius = radius


# Class to define hub object
class Hub(EnvironmentObject):
    def __init__(self, id=1, location=(0, 0), radius=20):
        super().__init__(id, location, radius)
        self.carryable = False
        self.dropable = True


# Class to define site object
class Sites(EnvironmentObject):
    def __init__(self, id=1, location=(0, 0), radius=20, q_value=0.5):
        super().__init__(id, location, radius)
        self.q_value = q_value
        self.food_unit = self.q_value * 1000
        self.carryable = False


# Class to define obstacle
class Obstacles(EnvironmentObject):
    def __init__(self, id=1, location=(0, 0), radius=20):
        super().__init__(id, location, radius)
        self.potential_field = None
        self.carrable = False
        self.dropable = True


# Class to define carryable property
class Carryable(EnvironmentObject):
    def __init__(self, id=1, location=(0, 0), radius=20):
        super().__init__(id, location, radius)
        # Carryable boolen value
        self.carryable = True
        self.weight = int(self.radius / 2)
        self.motion = False
        self.agents = []

    def calc_relative_weight(self):
        relative_weight = weight
        for agent in self.agents:
            relative_weight -= agent.capacity        

# Class to define communication
class Communication(EnvironmentObject):
    def __init__(self, id=1, location=(0, 0), radius=20, object_to_communicate=None):
        super().__init__(id, location, radius)
        # Communication parameters for signal
        self.communicated_object = object_to_communicate
        self.communicated_location = self.communicated_object.location


# Class to define signal
class Signal(Communication):
    def __init__(self, id=1, location=(0, 0), radius=20, object_to_communicate=None):
        super().__init__(id, None, radius, object_to_communicate)


# Class to define Cue
class Cue(Communication):
    def __init__(self, id=1, location=(0, 0), radius=20, object_to_communicate=None):
        super().__init__(id, location, radius, object_to_communicate)


# Class to define Traps
class Traps(EnvironmentObject):
    def __init__(self, id=1, location=(0, 0), radius=20):
        super().__init__(id, location, radius)
        self.carryable = False


# Class to define Food
class Food(Carryable):
    def __init__(self, id=1, location=(0, 0), radius=2):
        super().__init__(id, location, radius)


# Class to define Derbis
class Derbis(Carryable):
    def __init__(self, id=1, location=(0, 0), radius=2, weight=5):
        super().__init__(id, location, radius)

            
# Class to define Rules
class Rules:
    def __init__(self, id=1, json_data=None):
        self.id = id
        # self.currstate = Cu(id,json_data[0]['currstate'])
        self.currstate = Currstate(id, json_data[0]['currstate'])
        self.luggage = Luggage(id, json_data[1]['luggage'])
        self.movement = Movement(id, json_data[2]['movement'])
        self.communicate = None


# Class to define current state
class Currstate:
    def __init__(self, id=id, state_data=None):
        self.currstate_data = state_data
        self.curr_carry = self.currstate_data['carry']
        self.curr_drop = self.currstate_data['drop']
        self.curr_move = self.currstate_data['move']


# Class to define Luggage
class Luggage:
    def __init__(self, id=id, luggage_data=None):
        self.luggage_data = luggage_data
        self.carry = Carry(id, self.luggage_data['carry'])
        self.drop = Drop(id, self.luggage_data['drop'])


# Class to define Movement
class Movement:
    def __init__(self, id=id, movement_data=None):
        self.movement_data = movement_data
        self.conditions = self.movement_data['conditions']
        self.move = self.movement_data['move']
        self.orientation = self.movement_data['orientation']

        # self.carry = Carry(id,self.movement_data['carry'])
        # self.drop = Drop(id,self.movement_data['drop'])
        # print (self.conditions,self.move,self.orientation)


# Class to define Carry
class Carry:
    def __init__(self, id=id, carry_data=None):
        self.carry_data = carry_data
        self.conditions = self.carry_data[0]
        self.action = self.carry_data[1]


# Class to define Drop
class Drop:
    def __init__(self, id=id, drop_data=None):
        self.drop_data = drop_data
        self.conditions = self.drop_data[0]
        self.action = self.drop_data[1]
