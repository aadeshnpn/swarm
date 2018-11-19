"""Inherited model class."""

from swarms.lib.model import Model
from swarms.lib.time import SimultaneousActivation
# RandomActivation, StagedActivation
from swarms.lib.space import Grid
from swarms.utils.jsonhandler import JsonData
from swarms.utils.results import Best, Experiment
from swarms.utils.db import Connect
from agent import LearningAgent, ExecutingAgent
from swarms.lib.objects import (    # noqa : F401
    Hub, Sites, Food, Debris, Obstacles)
import os
import imp
import datetime
import numpy as np

filename = os.path.join(imp.find_module("swarms")[1] + "/utils/world.json")


class ForagingModel(Model):
    """A environemnt to model foraging environment."""

    def __init__(
            self, N, width, height, grid=10, iter=100000,
            seed=None, name='SForaging'):
        """Initialize the attributes."""
        if seed is None:
            super(ForagingModel, self).__init__(seed=None)
        else:
            super(ForagingModel, self).__init__(seed)

        # Create a unique experiment id
        self.runid = datetime.datetime.now().timestamp()
        self.runid = str(self.runid).replace('.', '')

        # Create the experiment folder
        self.pname = '/'.join(
            os.getcwd().split('/')[:-2]
            ) + '/results/' + self.runid + '-' + str(iter) + name

        # Define some parameters to count the step
        self.stepcnt = 1
        self.iter = iter

        # Create db connection
        connect = Connect('swarm', 'swarm', 'swarm', 'localhost')
        self.connect = connect.tns_connect()

        # Fill out the experiment table
        self.experiment = Experiment(
            self.connect, self.runid, N, seed, name,
            iter, width, height, grid)
        self.experiment.insert_experiment()

        # Get the primary key of the experiment table for future use
        self.sn = self.experiment.sn

        # Create a folder to store results
        os.mkdir(self.pname)

        # Number of agents
        self.num_agents = N
        # Environmental grid size
        self.grid = Grid(width, height, grid)
        # Schedular to active agents
        self.schedule = SimultaneousActivation(self)
        # Empty list of hold the agents
        self.agents = []

    def create_agents(self, random_init=True, phenotypes=None):
        """Initialize agents in the environment."""
        # This is abstract class. Each class inherting this
        # must define this on its own
        pass

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
        # First create the agents in the environment
        for name in jsondata.keys():
            obj = eval(name.capitalize())
            self.render.objects[name] = self.create_environment_object(
                jsondata, obj)

        self.hub = self.render.objects['hub'][0]
        self.total_food_units = 0
        try:
            self.site = self.render.objects['sites'][0]
            for i in range(self.num_agents * 1):
                f = Food(
                    i, location=self.site.location, radius=self.site.radius)
                f.agent_name = None
                self.grid.add_object_to_grid(f.location, f)
                self.total_food_units += f.weight
        except KeyError:
            pass

    def step(self):
        """Step through the environment."""
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
            self.pname, self.connect, self.sn, 1, 'MEAN', self.stepcnt,
            beta, np.mean(fittest), np.mean(diversity), np.mean(exploration),
            np.mean(foraging), "None"
            )
        mean.save()

        std = Best(
            self.pname, self.connect, self.sn, 1, 'STD', self.stepcnt, beta,
            np.std(fittest), np.std(diversity), np.std(exploration),
            np.std(foraging), "None"
            )
        std.save()

        # Compute best agent for each fitness
        self.best_agents(diversity, beta, "DIVERSE")
        self.best_agents(exploration, beta, "EXPLORE")
        self.best_agents(foraging, beta, "FORGE")
        self.best_agents(fittest, beta, "OVERALL")
        return np.argmax(foraging)

    def best_agents(self, data, beta, header):
        """Find the best agents in each category."""
        idx = np.argmax(data)
        dfitness = self.agents[idx].diversity_fitness
        ofitness = self.agents[idx].individual[0].fitness
        ffitness = self.agents[idx].food_collected
        efitness = self.agents[idx].exploration_fitness()
        phenotype = self.agents[idx].individual[0].phenotype

        best_agent = Best(
            self.pname, self.connect, self.sn, idx, header, self.stepcnt, beta,
            ofitness, dfitness, efitness, ffitness, phenotype
        )

        best_agent.save()

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

    def foraging_percent(self):
        """Compute the percent of the total food in the hub."""
        grid = self.grid
        hub_loc = self.hub.location
        neighbours = grid.get_neighborhood(hub_loc, 10)
        food_objects = grid.get_objects_from_list_of_grid('Food', neighbours)
        total_food_weights = sum([food.weight for food in food_objects])
        return ((total_food_weights * 1.0) / self.total_food_units) * 100


class EvolveModel(ForagingModel):
    """A environemnt to model swarms."""

    def __init__(
            self, N, width, height, grid=10, iter=100000,
            seed=None, name="EvoSForge"):
        """Initialize the attributes."""
        super(EvolveModel, self).__init__(
            N, width, height, grid, iter, seed, name)

    def create_agents(self, random_init=True, phenotypes=None):
        """Initialize agents in the environment."""
        # Create agents
        for i in range(self.num_agents):
            a = LearningAgent(i, self)
            # Add agent to the scheduler
            self.schedule.add(a)
            # Add the hub to agents memory
            a.shared_content['Hub'] = {self.hub}
            # First intitialize the Genetic algorithm. Then BT
            a.init_evolution_algo()
            # Initialize the BT. Since the agents are evolutionary
            # the bt will be random
            a.construct_bt()
            # Add the agent to a random grid cell
            x = self.random.randint(
                -self.grid.width / 2, self.grid.width / 2)
            y = self.random.randint(
                -self.grid.height / 2, self.grid.height / 2)

            a.location = (x, y)
            self.grid.add_object_to_grid((x, y), a)
            a.operation_threshold = 2  # self.num_agents // 10
            self.agents.append(a)

    def behavior_sampling(self, method='ratio', ratio_value=0.4):
        """Extract phenotype of the learning agents.

        Sort the agents based on the overall fitness and then based on the
        method extract phenotype of the agents.
        Method can take {'ratio','higest','sample'}
        """
        sorted_agents = sorted(
            self.agents, key=lambda x: x.individual[0].fitness, reverse=True)

        if method == 'ratio':
            upper_bound = ratio_value * self.num_agents
            selected_agents = self.agents[0:int(upper_bound)]
            selected_phenotype = [
                agent.individual[0].phenotype for agent in selected_agents]
            return selected_phenotype
        else:
            return [sorted_agents[0].individual[0].phenotype]

    def step(self):
        """Step through the environment."""
        # Gather info to plot the graph
        self.gather_info()

        # Next step
        self.schedule.step()

        # Increment the step count
        self.stepcnt += 1


class ValidationModel(ForagingModel):
    """A environemnt to validate swarm behaviors."""

    def __init__(
            self, N, width, height, grid=10, iter=100000,
            seed=None, name="ValidateSForge"):
        """Initialize the attributes."""
        super(ValidationModel, self).__init__(
            N, width, height, grid, iter, seed, name)

    def create_agents(self, random_init=False, phenotypes=None):
        """Initialize agents in the environment."""
        # Variable to tell how many agents will have the same phenotype
        bound = np.ceil((self.num_agents * 1.0) / len(phenotypes))
        j = 0
        # Create agents
        for i in range(self.num_agents):
            # print (i, j, self.xmlstrings[j])
            a = ExecutingAgent(i, self, xmlstring=phenotypes[j])
            # Add the agent to schedular list
            self.schedule.add(a)
            # Add the hub to agents memory
            a.shared_content['Hub'] = {self.hub}
            # Initialize the BT. Since the agents are normal agents just
            # use the phenotype
            a.construct_bt()

            if random_init:
                # Add the agent to a random grid cell
                x = self.random.randint(
                    -self.grid.width / 2, self.grid.width / 2)
                y = self.random.randint(
                    -self.grid.height / 2, self.grid.height / 2)
            try:
                x, y = self.hub.location
            except AttributeError:
                x, y = 0, 0

            a.location = (x, y)
            self.grid.add_object_to_grid((x, y), a)
            a.operation_threshold = 2  # self.num_agents // 10
            self.agents.append(a)

            if (i + 1) % bound == 0:
                j += 1


class TestModel(ForagingModel):
    """A environemnt to test swarm behavior performance."""

    def __init__(
            self, N, width, height, grid=10, iter=100000,
            seed=None, name="TestSForge"):
        """Initialize the attributes."""
        super(TestModel, self).__init__(
            N, width, height, grid, iter, seed, name)

    def create_agents(self, random_init=False, phenotypes=None):
        """Initialize agents in the environment."""
        # Variable to tell how many agents will have the same phenotype
        bound = np.ceil((self.num_agents * 1.0) / len(phenotypes))
        j = 0
        # Create agents
        for i in range(self.num_agents):
            # print (i, j, self.xmlstrings[j])
            a = ExecutingAgent(i, self, xmlstring=phenotypes[j])
            self.schedule.add(a)
            # Add the hub to agents memory
            a.shared_content['Hub'] = {self.hub}
            # Initialize the BT. Since the agents are normal agents just
            # use the phenotype
            a.construct_bt()
            if random_init:
                # Add the agent to a random grid cell
                x = self.random.randint(
                    -self.grid.width / 2, self.grid.width / 2)
                y = self.random.randint(
                    -self.grid.height / 2, self.grid.height / 2)
            try:
                x, y = self.hub.location
            except AttributeError:
                x, y = 0, 0

            a.location = (x, y)
            self.grid.add_object_to_grid((x, y), a)
            a.operation_threshold = 2  # self.num_agents // 10
            self.agents.append(a)

            if (i + 1) % bound == 0:
                j += 1
