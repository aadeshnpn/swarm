from lib.model import Model
from lib.time import SimultaneousActivation  # RandomActivation, StagedActivation
from lib.space import Grid
from swarms.utils.jsonhandler import JsonData
from swarms.objects import Hub
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

    def build_environment_from_json(self):
        jsondata = JsonData.load_json_file(filename)
        i = 0
        render_jsondata = JsonData()
        render_jsondata.hub = []
        for jhub in jsondata["hub"]:
            location = (jhub["x"], jhub["y"])
            hub = Hub(i, location, jhub["radius"])
            self.grid.add_object_to_grid(location, hub)
            render_jsondata.hub.append(hub)

    def step(self):
        self.schedule.step()
