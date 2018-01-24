from lib.agent import Agent

# from swarms.sbehaviors import NeighbourObjects, ShareMoney, Move, HasMoney
import numpy as np
# import py_trees


class SwarmAgent(Agent):
    """ An minimalistic swarm agent """
    def __init__(self, name, model):
        super().__init__(name, model)
        self.location = ()

        self.direction = model.random.rand() * (2 * np.pi)
        self.speed = 2
        self.radius = 3

        self.capacity = 10
        self.attached_objects = []
        self.moveable = True
        # behaviour_tree.setup(15)
        """
        root = py_trees.composites.Sequence("Sequence")
        low = Move('4')
        low.setup(0, self)        
        higest = HasMoney('1')
        higest.setup(0, self)
        high = NeighbourObjects('2')
        high.setup(0, self, 'SwarmAgent')
        med = ShareMoney('3')
        med.setup(0, self)

        root.add_children([low, higest, high, med])
        self.behaviour_tree = py_trees.trees.BehaviourTree(root)
        """
        self.shared_contents = dict()

    # New Agent methods for behavior based robotics
    def sense(self):
        pass

    def plan(self):
        pass

    # Make necessary Changes
    def step(self):
        # if self.wealth > 0:
        #    self.give_money()
        # self.behaviour_tree.tick()
        self.move()
        if self.wealth > 0:
            self.give_money()

    # Applies the changes
    def advance(self):
        # self.move()
        pass

    """
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
    """