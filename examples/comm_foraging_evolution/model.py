"""Inherited model class."""

from swarms.lib.model import Model
from swarms.lib.time import SimultaneousActivation
# RandomActivation, StagedActivation
from swarms.lib.space import Grid
from swarms.utils.jsonhandler import JsonData
from agent import SwarmAgent
from swarms.lib.objects import Hub, Sites, Food, Derbis, Obstacles
import os, imp

#filename = os.path.join(
#    "/home/aadeshnpn/Documents/BYU/hcmi/swarm/swarms/" + "utils/world.json")
filename = os.path.join(imp.find_module("swarms")[1] + "/utils/world.json")


class EnvironmentModel(Model):
    """A environemnt to model swarms."""

    def __init__(self, N, width, height, grid=10, seed=None):
        """Initialize the attributes."""
        if seed is None:
            super(EnvironmentModel, self).__init__(seed=None)
        else:
            super(EnvironmentModel, self).__init__(seed)

        self.num_agents = N

        self.grid = Grid(width, height, grid)

        self.schedule = SimultaneousActivation(self)

        self.site = Sites(id=1, location=(29, 29), radius=5, q_value=0.5)

        self.grid.add_object_to_grid(self.site.location, self.site)

        # self.hub = Hub(id=1, location=(0, 0), radius=11)

        # self.grid.add_object_to_grid(self.hub.location, self.hub)

        self.agents = []

        # Create agents
        for i in range(self.num_agents):
            a = SwarmAgent(i, self)
            self.schedule.add(a)
            # Add the agent to a random grid cell
            x = self.random.randint(
                -self.grid.width / 2, self.grid.width / 2)
            # x = 0
            y = self.random.randint(
                -self.grid.height / 2, self.grid.height / 2)
            # y = 0

            a.location = (x, y)
            self.grid.add_object_to_grid((x, y), a)
            a.operation_threshold = 2  # self.num_agents // 10
            self.agents.append(a)

        # Add equal number of food source
        # for i in range(20):
        #    f = Food(i, location=(-29, -29), radius=5)
        #    self.grid.add_object_to_grid(f.location, f)
            # print (i,x,y)

    def create_environment_object(self, jsondata, obj):
        """Create env from jsondata."""
        name = obj.__name__.lower()
        temp_list = []
        i = 0
        for json_object in jsondata[name]:
            location = (json_object["x"], json_object["y"])
            if "q_value" in json_object:
                temp_obj = obj(
                    i, location, json_object["radius"], q_value=json_object[
                        "q_value"])
            else:
                temp_obj = obj(i, location, json_object["radius"])

            self.grid.add_object_to_grid(location, temp_obj)
            temp_list.append(temp_obj)
            i += 1
        return temp_list

    def build_environment_from_json(self):
        """Build env from jsondata."""
        jsondata = JsonData.load_json_file(filename)
        # Create a instance of JsonData to store object that
        # needs to be sent to UI
        self.render = JsonData()
        self.render.objects = {}

        for name in jsondata.keys():
            obj = eval(name.capitalize())
            self.render.objects[name] = self.create_environment_object(
                jsondata, obj)

        self.hub = self.render.objects['hub'][0]
        try:
            self.site = self.render.objects['sites'][0]
            for i in range(50):
                f = Food(i, location=self.site.location, radius=self.site.radius)
                f.agent_name = None
                self.grid.add_object_to_grid(f.location, f)
        except KeyError:
            pass

    def step(self):
        """Step through the environment."""
        self.schedule.step()

    def find_higest_performer(self):
        """Find the best agent."""
        fitness = self.agents[0].individual[0].fitness
        fittest = self.agents[0]
        for agent in self.agents:
            if agent.individual[0].fitness > fitness:
                fittest = agent
        return fittest

    def find_higest_food_collector(self):
        fitness = self.agents[0].food_collected
        fittest = self.agents[0]
        for agent in self.agents:
            if agent.food_collected > fitness:
                fittest = agent
        return fittest

    def detect_food_moved(self):
        grid = self.grid
        food_loc = self.site.location
        neighbours = grid.get_neighborhood(food_loc, 10)
        food_objects = grid.get_objects_from_list_of_grid('Food', neighbours)

        # print (food_objects)
        return food_objects
