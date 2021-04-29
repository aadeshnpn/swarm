"""Inherited model class."""

from swarms.lib.model import Model
from swarms.lib.time import SimultaneousActivation
# RandomActivation, StagedActivation
from swarms.lib.space import Grid
from swarms.utils.jsonhandler import JsonData
from swarms.utils.results import Experiment
from swarms.utils.db import Connect
from swarms.utils.results import Best
from simagent import SimForgAgentWithout, SimForgAgentWith, EvolAgent, SimAgent
from swarms.lib.objects import Hub, Sites, Food, Debris, Obstacles, Traps
import os
import pathlib
import imp
import datetime
import numpy as np
from swarms.utils.ui import UI

import pathlib

current_dir = pathlib.Path(__file__).parent
filename = os.path.join(
    str(current_dir) + "/world.json")



class SimForgModel(Model):
    """A environemnt to model swarms."""

    def __init__(
            self, N, width, height, grid=10, iter=100000,
            xmlstrings=None, seed=None, viewer=False, pname=None, agent=SimForgAgentWithout, expsite=None):
        """Initialize the attributes."""
        if seed is None:
            super(SimForgModel, self).__init__(seed=None)
        else:
            super(SimForgModel, self).__init__(seed)

        self.runid = datetime.datetime.now().strftime(
            "%s") + str(self.random.randint(1, 10000, 1)[0])

        if pname is None:
            self.pname = os.getcwd() + '/' + self.runid + "SForagingSimulation"
        else:
            self.pname = pname + '/' + self.runid + "SForagingSimulation"

        self.width = width
        self.height = height
        self.stepcnt = 1
        self.iter = iter
        self.xmlstrings = xmlstrings

        self.viewer = viewer
        self.agent = agent
        self.expsite = expsite
        # print('agent type', agent)
        # # Create db connection
        # try:
        #     connect = Connect('swarm', 'swarm', 'swarm', 'localhost')
        #     self.connect = connect.tns_connect()
        # except:
        #     pass
        self.connect = None
        # # # Fill out the experiment table
        # self.experiment = Experiment(
        #     self.connect, self.runid, N, seed, 'Simuation Single Foraging',
        #     iter, width, height, grid, phenotype=xmlstrings[0])

        # self.experiment.insert_experiment_simulation()

        # self.sn = self.experiment.sn
        self.sn = 1
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

        bound = np.ceil((self.num_agents * 1.0) / len(self.xmlstrings))

        j = 0
        # Create agents
        for i in range(self.num_agents):
            # print (i, j, self.xmlstrings[j])
            a = self.agent(i, self, xmlstring=self.xmlstrings[j])
            self.schedule.add(a)
            # Add the agent to a random grid cell
            # x = self.random.randint(
            #    -self.grid.width / 2, self.grid.width / 2)
            x = 0
            # y = self.random.randint(
            #    -self.grid.height / 2, self.grid.height / 2)
            y = 0

            a.location = (x, y)
            self.grid.add_object_to_grid((x, y), a)
            a.operation_threshold = 2  # self.num_agents // 10
            self.agents.append(a)

            if (i + 1) % bound == 0:
                j += 1

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
        self.obstacles = self.render.objects['obstacles'][0]
        # print(self.obstacles.passable)
        self.traps = self.render.objects['traps'][0]  

        # add site
        location = (self.expsite["x"], self.expsite["y"])
        self.site = Sites(
                0, location, self.expsite["radius"], q_value=self.expsite[
                    "q_value"])

        self.grid.add_object_to_grid(location, self.site)

        try:
            # self.site = self.render.objects['sites'][0]
            self.foods = []
            for i in range(self.num_agents * 1):
                f = Food(
                    i, location=self.site.location, radius=self.site.radius)
                f.agent_name = None
                self.grid.add_object_to_grid(f.location, f)
                self.foods.append(f)
        except KeyError:
            pass

        # Add trap
        # try:
        #     # x,y = self.site.location
        #     # x = self.random.randint(x+20, x+40)
        #     # y = self.random.randint(y+20, y+40)
        #     # # print(self.site.location, x, y)
        #     # self.traps = []
        #     # f = Traps(
        #     #     i, location=(x,y), radius=20)
        #     # f.agent_name = None
        #     # self.grid.add_object_to_grid(f.location, f)
        #     # self.traps.append(f)
        # except KeyError:
        #     pass

        if self.viewer:
            self.ui = UI(
                (self.width, self.height), [self.hub], self.agents,
                [self.site], food=self.foods)

    def step(self):
        """Step through the environment."""
        # Gather info from all the agents
        # self.gather_info()
        # Next step
        self.schedule.step()
        # Increment the step count
        self.stepcnt += 1
        if self.viewer:
            self.ui.step()

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

    def food_in_hub(self):
        """Find amount of food in hub."""
        grid = self.grid
        food_loc = self.hub.location
        neighbours = grid.get_neighborhood(food_loc, 10)
        food_objects = grid.get_objects_from_list_of_grid('Food', neighbours)
        return len(food_objects)

    def food_in_loc(self, loc):
        """Find amount of food in hub."""
        grid = self.grid
        neighbours = grid.get_neighborhood(loc, 10)
        food_objects = grid.get_objects_from_list_of_grid('Food', neighbours)
        return food_objects

    def no_agent_dead(self):
        grid = self.grid
        trap_loc = self.traps.location
        neighbours = grid.get_neighborhood(trap_loc, 10)
        agents = grid.get_objects_from_list_of_grid(type(self.agents[0]).__name__, neighbours)
        return sum([1 if a.dead else 0 for a in agents])     


class EvolModel(Model):
    """A environemnt to model swarms."""

    def __init__(
            self, N, width, height, grid=10, iter=100000, seed=None,
            expname='MSForaging', agent='EvolAgent', parm='res.txt'):
        """Initialize the attributes."""
        if seed is None:
            super(EvolModel, self).__init__(seed=None)
        else:
            super(EvolModel, self).__init__(seed)

        self.runid = datetime.datetime.now().strftime(
            "%s") + str(self.random.randint(1, 1000, 1)[0])

        # self.pname = '/'.join(os.getcwd().split('/')[:-2]) + '/results/' \
        #     + self.runid + expname
        self.pname = os.path.join('/tmp', 'swarm', 'data', 'experiments',str(N), agent, str(iter),  str(self.runid)+expname)
        pathlib.Path(self.pname).mkdir(parents=True, exist_ok=True)
        self.stepcnt = 1
        self.iter = iter
        self.top = None
        # Create db connection
        self.connect = None
        # connect = Connect('swarm', 'swarm', 'swarm', 'localhost')
        # self.connect = connect.tns_connect()

        # # Fill out the experiment table
        # self.experiment = Experiment(
        #     self.connect, self.runid, N, seed, expname,
        #     iter, width, height, grid)
        # self.experiment.insert_experiment()

        # self.sn = self.experiment.sn
        self.sn = 1
        # Create a folder to store results
        # os.mkdir(self.pname)

        self.num_agents = N

        self.grid = Grid(width, height, grid)

        self.schedule = SimultaneousActivation(self)

        self.agents = []
        self.parm = parm

        # Create agents
        for i in range(self.num_agents):
            a = eval(agent)(i, self)
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
        self.site = Sites(
                0, (30, -30), 10, q_value=0.9)

        self.grid.add_object_to_grid((30, -30), self.site)

        try:
            self.foods = []
            for site in [self.site]:
                # self.site = site  # self.render.objects['sites'][0]

                for i in range(self.num_agents):
                    f = Food(
                        i, location=self.site.location,
                        radius=self.site.radius)
                    f.agent_name = None
                    self.grid.add_object_to_grid(f.location, f)
                    self.foods.append(f)
        except KeyError:
            pass

    def step(self):
        """Step through the environment."""
        try:
            # Gather info from all the agents
            self.top = self.gather_info()
        except FloatingPointError:
            pass
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

    def food_in_loc(self, loc):
        """Find amount of food in hub."""
        grid = self.grid
        neighbours = grid.get_neighborhood(loc, 10)
        food_objects = grid.get_objects_from_list_of_grid('Food', neighbours)
        return food_objects

    def behavior_sampling_objects(self, method='ratio', ratio_value=0.2):
        """Extract phenotype of the learning agents based on the objects.

        Sort the phenotye based on the overall fitness and then based on the
        method extract phenotype of the agents.
        Method can take {'ratio','higest','sample'}
        """
        # sorted_agents = sorted(
        #    self.agents, key=lambda x: x.individual[0].fitness, reverse=True)
        # phenotypes = dict()
        # Get the phenotypes collected from the agent
        phenotypes = self.phenotype_attached_objects()

        for agent in self.agents:
            phenotypes = {**agent.phenotypes, **phenotypes}

        # Sort the phenotypes
        phenotypes, _ = zip(
            *sorted(phenotypes.items(), key=lambda x: (
                x[1]), reverse=True))

        if method == 'ratio':
            upper_bound = ratio_value * self.num_agents
            # selected_agents = self.agents[0:int(upper_bound)]
            # selected_phenotype = [
            #    agent.individual[0].phenotype for agent in selected_agents]
            selected_phenotype = list(phenotypes)[:int(upper_bound)]
            return selected_phenotype
        else:
            # return [sorted_agents[0].individual[0].phenotype]
            return phenotypes[0]

    def phenotype_attached_objects(self):
        """Extract phenotype from the objects."""
        # grid = self.grid
        # hub_loc = self.hub.location
        # neighbours = grid.get_neighborhood(hub_loc, 20)
        # food_objects = grid.get_objects_from_list_of_grid('Food', neighbours)
        phenotypes = dict()
        for food in self.foods:
            # phenotypes += list(food.phenotype.values())
            try:
                phenotypes = {**food.phenotype, ** phenotypes}
            except (AttributeError, ValueError):
                pass
        # print ('phenotypes for attached objects', phenotypes)
        return phenotypes  
              

class SimModel(Model):
    """A environemnt to model swarms."""

    def __init__(
            self, N, width, height, grid=10, iter=100000,
            xmlstrings=None, seed=None, viewer=False, pname=None,
            expname='MSFSimulation', agent='SimAgent'):
        """Initialize the attributes."""
        if seed is None:
            super(SimModel, self).__init__(seed=None)
        else:
            super(SimModel, self).__init__(seed)

        self.runid = datetime.datetime.now().strftime(
            "%s") + str(self.random.randint(1, 1000, 1)[0])

        # if pname is None:
        #     self.pname = os.getcwd() + '/' + self.runid + expname
        # else:
        #     self.pname = pname + '/' + self.runid + expname

        if pname is None:
            self.pname = os.path.join('/tmp', 'swarm', 'data', 'experiments',str(N), agent, str(iter), str(self.runid)+expname)
        else:
            self.pname = os.path.join(pname, str(self.runid) + expname)


        pathlib.Path(self.pname).mkdir(parents=True, exist_ok=True)
        self.width = width
        self.height = height
        self.stepcnt = 1
        self.iter = iter
        self.xmlstrings = xmlstrings
        # print(xmlstrings, type(xmlstrings))

        self.viewer = viewer

        # Create db connection
        self.connect = None
        # connect = Connect('swarm', 'swarm', 'swarm', 'localhost')
        # self.connect = connect.tns_connect()

        # # Fill out the experiment table
        # self.experiment = Experiment(
        #     self.connect, self.runid, N, seed, expname,
        #     iter, width, height, grid, phenotype=xmlstrings[0])

        # self.experiment.insert_experiment_simulation()

        # self.sn = self.experiment.sn
        self.sn = 1
        # # Create a folder to store results
        # os.mkdir(self.pname)

        self.num_agents = N

        self.grid = Grid(width, height, grid)

        self.schedule = SimultaneousActivation(self)

        self.agents = []

        bound = np.ceil((self.num_agents * 1.0) / len(self.xmlstrings))

        j = 0
        # Create agents
        for i in range(self.num_agents):
            # print (i, j, self.xmlstrings[j])
            a = eval(agent)(i, self, xmlstring=self.xmlstrings[j])
            self.schedule.add(a)
            # Add the agent to a random grid cell
            # x = self.random.randint(
            #    -self.grid.width / 2, self.grid.width / 2)
            x = 0
            # y = self.random.randint(
            #    -self.grid.height / 2, self.grid.height / 2)
            y = 0

            a.location = (x, y)
            self.grid.add_object_to_grid((x, y), a)
            a.operation_threshold = 2  # self.num_agents // 10
            self.agents.append(a)

            if (i + 1) % bound == 0:
                j += 1

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

        # add site
        # location = (self.expsite["x"], self.expsite["y"])
        self.site = Sites(
                0, (30, -30), 10, q_value=0.9)

        self.grid.add_object_to_grid((30, -30), self.site)

        try:
            self.foods = []
            # for site in self.render.objects['sites']:
            for site in [self.site]:                
                # self.site = site  # self.render.objects['sites'][0]

                for i in range(self.num_agents):

                    f = Food(
                        i, location=self.site.location,
                        radius=self.site.radius)
                    f.agent_name = None
                    self.grid.add_object_to_grid(f.location, f)
                    self.foods.append(f)
        except KeyError:
            pass

        if self.viewer:
            self.ui = UI(
                (self.width, self.height), [self.hub], self.agents,
                [self.site], food=self.foods)

    def step(self):
        """Step through the environment."""
        # Gather info from all the agents
        # self.gather_info()
        # Next step
        self.schedule.step()
        # Increment the step count
        self.stepcnt += 1
        if self.viewer:
            self.ui.step()

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

    def food_in_hub(self):
        """Find amount of food in hub."""
        grid = self.grid
        food_loc = self.hub.location
        neighbours = grid.get_neighborhood(food_loc, 10)
        food_objects = grid.get_objects_from_list_of_grid('Food', neighbours)
        return len(food_objects)

    def food_in_loc(self, loc):
        """Find amount of food in hub."""
        grid = self.grid
        neighbours = grid.get_neighborhood(loc, 10)
        food_objects = grid.get_objects_from_list_of_grid('Food', neighbours)
        return food_objects
