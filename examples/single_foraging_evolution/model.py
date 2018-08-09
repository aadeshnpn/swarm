"""Inherited model class."""

from swarms.lib.model import Model
from swarms.lib.time import SimultaneousActivation
# RandomActivation, StagedActivation
from swarms.lib.space import Grid
from swarms.utils.jsonhandler import JsonData
from swarms.utils.results import Best
from agent import SwarmAgent
from swarms.lib.objects import Hub, Sites, Food, Derbis, Obstacles
import os
import imp
import datetime
import numpy as np

filename = os.path.join(imp.find_module("swarms")[1] + "/utils/world.json")


class EnvironmentModel(Model):
    """A environemnt to model swarms."""

    def __init__(self, N, width, height, grid=10, iter=100000, seed=None):
        """Initialize the attributes."""
        if seed is None:
            super(EnvironmentModel, self).__init__(seed=None)
        else:
            super(EnvironmentModel, self).__init__(seed)

        self.pname = os.getcwd() + '/' + datetime.datetime.now().strftime(
            "%s") + "SForaging"

        self.stepcnt = 1
        self.iter = iter
        # Create a folder to store results
        os.mkdir(self.pname)

        self.num_agents = N

        self.grid = Grid(width, height, grid)

        self.schedule = SimultaneousActivation(self)

        # self.site = Sites(id=1, location=(5, 5), radius=11, q_value=0.5)

        # self.grid.add_object_to_grid(self.site.location, self.site)

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
                f = Food(
                    i, location=self.site.location, radius=self.site.radius)
                f.agent_name = None
                self.grid.add_object_to_grid(f.location, f)
        except KeyError:
            pass

    def step(self):
        """Step through the environment."""
        # Gather info from all the agents
        self.gather_info()
        # Next step
        self.schedule.step()
        # Increment the step count
        self.stepcnt += 1

    def gather_info(self):
        """Gather information from all the agents."""
        diversity = np.ones(len(self.agents))
        exploration = np.ones(len(self.agents))
        foraging = np.ones(len(self.agents))
        fittest = np.ones(len(self.agents))
        for id in range(len(self.agents)):
            diversity[id] = self.agents[id].diversity_fitness
            exploration[id] = self.agents[id].exploration_fitness()
            foraging[id] = self.agents[id].food_collected
            fittest[id] = self.agents[id].individual[0].fitness
        beta = self.agents[-1].beta

        mean = Best(
            self.pname, "", 'MEAN', self.stepcnt, beta, np.mean(fittest),
            np.mean(diversity), np.mean(exploration), np.mean(foraging),
            ""
            )
        mean.save_to_file()

        std = Best(
            self.pname, "", 'STD', self.stepcnt, beta, np.std(fittest),
            np.std(diversity), np.std(exploration), np.std(foraging), ""
            )
        std.save_to_file()

        # Compute best agent for each fitness
        self.best_agents(diversity, beta, "DIVERSE")
        self.best_agents(exploration, beta, "EXPLORE")
        self.best_agents(foraging, beta, "FORGE")
        self.best_agents(fittest, beta, "OVERALL")

    def best_agents(self, data, beta, header):
        """Find the best agents in each category."""
        idx = np.argmax(data)
        dfitness = self.agents[idx].diversity_fitness
        ofitness = self.agents[idx].individual[0].fitness
        ffitness = self.agents[idx].food_collected
        efitness = self.agents[idx].exploration_fitness()
        phenotype = self.agents[idx].individual[0].phenotype

        best_agent = Best(
            self.pname, idx, header, self.stepcnt, beta, ofitness,
            dfitness, efitness, ffitness, phenotype
        )

        best_agent.save_to_file()

    def find_higest_performer(self):
        """Find the best agent."""
        fitness = self.agents[0].individual[0].fitness
        fittest = self.agents[0]
        for agent in self.agents:
            if agent.individual[0].fitness > fitness:
                fittest = agent
        return fittest

    def find_higest_food_collector(self):
        """Find the best agent to collect food."""
        fitness = self.agents[0].food_collected
        fittest = self.agents[0]
        for agent in self.agents:
            if agent.food_collected > fitness:
                fittest = agent
        return fittest

    def detect_food_moved(self):
        """Detect food moved."""
        grid = self.grid
        food_loc = self.site.location
        neighbours = grid.get_neighborhood(food_loc, 10)
        food_objects = grid.get_objects_from_list_of_grid('Food', neighbours)

        # print (food_objects)
        return food_objects
