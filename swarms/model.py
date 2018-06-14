"""Inherited model class."""

from swarms.lib.model import Model
from swarms.lib.time import SimultaneousActivation
# RandomActivation, StagedActivation
from swarms.lib.space import Grid
from swarms.utils.jsonhandler import JsonData
from swarms.agent import SwarmAgent
from swarms.objects import Hub, Sites, Obstacles, Traps, Derbis
import numpy as np
import os

filename = os.path.join(
    "/home/aadeshnpn/Documents/BYU/hcmi/swarm/swarms/" + "utils/world.json")


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

        self.build_environment_from_json()

        self.agents = []
        for i in range(self.num_agents):
            a = SwarmAgent(i, self)
            self.schedule.add(a)

            # Add the agent to a random grid cell
            # x = self.random.randint(
            #    -self.grid.width / 2, self.grid.width / 2)
            # y = self.random.randint(
            #    -self.grid.height / 2, self.grid.height / 2)
            x = -350 + np.random.randint(-20, 20)
            y = -350 + np.random.randint(-20, 20)
            a.location = (x, y)
            a.direction = -2.3561944901923448
            self.grid.add_object_to_grid((x, y), a)
            self.agents.append(a)
            print (i,x,y)

    def create_environment_object(self, jsondata, obj):
        """Create env from jsondata."""
        name = obj.__name__.lower()
        temp_list = []
        i = 0
        for json_object in jsondata[name]:
            location = (json_object["x"], json_object["y"])
            if "q_value" in json_object:
                temp_obj = obj(i, location, json_object["radius"], q_value=json_object["q_value"])
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
        self.render_jsondata = JsonData()
        self.render_jsondata.objects = {}

        for name in jsondata.keys():
            obj = eval(name.capitalize())
            self.render_jsondata.objects[name] = self.create_environment_object(jsondata, obj)

        self.hub = self.render_jsondata.objects['hub'][0]

    def step(self):
        """Step through the environment."""
        self.schedule.step()
