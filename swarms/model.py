from lib.model import Model
from lib.time import SimultaneousActivation  # RandomActivation, StagedActivation
from lib.space import Grid
from swarms.utils.jsonhandler import JsonData
from swarms.objects import Hub, Sites
from swarms.agent import SwarmAgent

filename = "swarms/utils/world.json"


class EnvironmentModel(Model):
    """ A environemnt to model swarms """
    def __init__(self, N, width, height, grid=10, seed=None):

        if seed is None:
            super(EnvironmentModel, self).__init__(seed=None)
        else:
            super(EnvironmentModel, self).__init__(seed)

        self.num_agents = N

        self.grid = Grid(width, height, grid)

        self.schedule = SimultaneousActivation(self)

        self.build_environment_from_json()

        for i in range(self.num_agents):
            a = SwarmAgent(i, self)
            self.schedule.add(a)

            # Add the agent to a random grid cell
            x = self.random.randint(-self.grid.width / 2, self.grid.width / 2)
            y = self.random.randint(-self.grid.height / 2, self.grid.height / 2)

            a.location = (x, y)
            self.grid.add_object_to_grid((x, y), a)

    def create_environment_object(self, jsondata, obj):
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
            i += 0
        return temp_list

    def build_environment_from_json(self):
        jsondata = JsonData.load_json_file(filename)

        # Create a instance of JsonData to store object that needs to be sent to UI
        self.render_jsondata = JsonData()

        self.render_jsondata.objects = {}
        for obj in [Hub, Sites]:
            name = obj.__name__.lower()
            self.render_jsondata.objects[name] = self.create_environment_object(jsondata, obj)
        print(self.render_jsondata.objects)
        exit()
        """
        # Create hub from json data
        self.render_jsondata.hubs = []
        for jhub in jsondata["hub"]:
            location = (jhub["x"], jhub["y"])
            hub = Hub(i, location, jhub["radius"])
            self.grid.add_object_to_grid(location, hub)
            self.render_jsondata.hub.append(hub)
            i += 1

        # Create site from json data
        render_jsondata.sites = []
        for jsite in jsondata["site"]:
            location = (jsite["x"], jsite["y"])
            site = Sites(i, location, jsite["radius"])
            self.grid.add_object_to_grid(location, site)
            render_jsondata.sites.append(site)
            i += 1
        """

    def step(self):
        self.schedule.step()
