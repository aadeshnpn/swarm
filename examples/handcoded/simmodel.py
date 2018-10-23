"""Inherited model class."""

from swarms.lib.model import Model
from swarms.lib.time import SimultaneousActivation
# RandomActivation, StagedActivation
from swarms.lib.space import Grid
from swarms.utils.jsonhandler import JsonData
from swarms.utils.results import Experiment
from swarms.utils.db import Connect
from simagent import SimForgAgent, SimCTAgent, SimNMAgent
from swarms.lib.objects import Hub, Sites, Food, Debris, Obstacles
import os
import imp
import datetime
import numpy as np
from swarms.utils.ui import UI

filename = os.path.join(imp.find_module("swarms")[1] + "/utils/world.json")


class SimForgModel(Model):
    """A environemnt to model swarms."""

    def __init__(
            self, N, width, height, grid=10, iter=100000,
            xmlstrings=None, seed=None, viewer=False, pname=None):
        """Initialize the attributes."""
        if seed is None:
            super(SimForgModel, self).__init__(seed=None)
        else:
            super(SimForgModel, self).__init__(seed)

        self.runid = datetime.datetime.now().strftime(
            "%s") + str(self.random.randint(1, 1000, 1)[0])

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

        # Create db connection
        connect = Connect('swarm', 'swarm', 'swarm', 'localhost')
        self.connect = connect.tns_connect()

        # Fill out the experiment table
        self.experiment = Experiment(
            self.connect, self.runid, N, seed, 'Simuation Single Foraging',
            iter, width, height, grid, phenotype=xmlstrings[0])

        self.experiment.insert_experiment_simulation()

        self.sn = self.experiment.sn

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
            a = SimForgAgent(i, self, xmlstring=self.xmlstrings[j])
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
        try:
            self.site = self.render.objects['sites'][0]
            self.foods = []
            for i in range(self.num_agents * 1):
                f = Food(
                    i, location=self.site.location, radius=self.site.radius)
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


class SimCTModel(Model):
    """A environemnt to model swarms."""

    def __init__(
            self, N, width, height, grid=10, iter=100000,
            xmlstrings=None, seed=None, viewer=False, pname=None,
            expname='COTSimulation', agent='SimCTAgent'):
        """Initialize the attributes."""
        if seed is None:
            super(SimCTModel, self).__init__(seed=None)
        else:
            super(SimCTModel, self).__init__(seed)

        self.runid = datetime.datetime.now().strftime(
            "%s") + str(self.random.randint(1, 1000, 1)[0])

        if pname is None:
            self.pname = os.getcwd() + '/' + self.runid + expname
        else:
            self.pname = pname + '/' + self.runid + expname

        self.width = width
        self.height = height
        self.stepcnt = 1
        self.iter = iter
        self.xmlstrings = xmlstrings

        self.viewer = viewer

        # Create db connection
        connect = Connect('swarm', 'swarm', 'swarm', 'localhost')
        self.connect = connect.tns_connect()

        # Fill out the experiment table
        self.experiment = Experiment(
            self.connect, self.runid, N, seed, expname,
            iter, width, height, grid, phenotype=xmlstrings[0])

        self.experiment.insert_experiment_simulation()

        self.sn = self.experiment.sn

        # Create a folder to store results
        os.mkdir(self.pname)

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
        try:
            self.foods = []
            self.site = self.render.objects['sites'][0]
            food_radius = self.random.randint(20, 30)
            for i in range(self.num_agents):
                f = Food(
                    i, location=self.site.location,
                    radius=food_radius)
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


class SimNMModel(Model):
    """A environemnt to model swarms."""

    def __init__(
            self, N, width, height, grid=10, iter=100000,
            xmlstrings=None, seed=None, viewer=False, pname=None,
            expname='NMSimulation', agent='SimNMAgent'):
        """Initialize the attributes."""
        if seed is None:
            super(SimNMModel, self).__init__(seed=None)
        else:
            super(SimNMModel, self).__init__(seed)

        # self.runid = datetime.datetime.now().strftime(
        #    "%s") + str(self.random.randint(1, 1000, 1)[0])

        self.runid = datetime.datetime.now().timestamp()
        self.runid = str(self.runid).replace('.', '')

        if pname is None:
            self.pname = os.getcwd() + '/' + self.runid + expname
        else:
            self.pname = pname + '/' + self.runid + expname

        self.width = width
        self.height = height
        self.stepcnt = 1
        self.iter = iter
        self.xmlstrings = xmlstrings

        self.viewer = viewer

        # Create db connection
        connect = Connect('swarm', 'swarm', 'swarm', 'localhost')
        self.connect = connect.tns_connect()

        # Fill out the experiment table
        self.experiment = Experiment(
            self.connect, self.runid, N, seed, expname,
            iter, width, height, grid, phenotype=xmlstrings[0])

        self.experiment.insert_experiment_simulation()

        self.sn = self.experiment.sn

        # Create a folder to store results
        os.mkdir(self.pname)

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
        try:
            self.debris = []
            for i in range(self.num_agents):
                dx, dy = self.random.randint(1, 10, 2)
                dx = self.hub.location[0] + dx
                dy = self.hub.location[1] + dy
                d = Debris(
                    i, location=(dx, dy),
                    radius=5)
                d.agent_name = None
                self.grid.add_object_to_grid(d.location, d)
                self.debris.append(d)
        except KeyError:
            pass

        # Create a place for the agents to drop the derbis
        try:
            self.obstacles = []
            for i in range(1):
                dx, dy = self.random.randint(5, 10, 2)
                dx = self.hub.location[0] + 25 + dx
                dy = self.hub.location[1] + 25 + dy
                o = Obstacles(id=i, location=(dx, dy), radius=10)
                self.grid.add_object_to_grid(o.location, o)
                self.obstacles.append(o)
        except AttributeError:
            pass

        if self.viewer:
            self.ui = UI(
                (self.width, self.height), [self.hub], self.agents,
                [], food=[], derbis=self.debris)

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

    def find_higest_debris_collector(self):
        """Find the best agent to collect debris."""
        fitness = self.agents[0].debris_collected
        fittest = self.agents[0]
        for agent in self.agents:
            if agent.food_collected > fitness:
                fittest = agent
        return fittest

    def detect_debris_moved(self):
        """Detect debris moved."""
        grid = self.grid
        debris_loc = self.hub.location
        neighbours = grid.get_neighborhood(debris_loc, 30)
        debris_objects = grid.get_objects_from_list_of_grid(
            'Debris', neighbours)

        return debris_objects

    def debris_around_hub(self):
        """Find amount of debris around hub."""
        grid = self.grid
        food_loc = self.hub.location
        neighbours = grid.get_neighborhood(food_loc, 20)
        food_objects = grid.get_objects_from_list_of_grid(
            'Debris', neighbours)
        return len(food_objects)

    def debris_in_loc(self, loc):
        """Find amount of debris in a location."""
        grid = self.grid
        neighbours = grid.get_neighborhood(loc, 10)
        debris_objects = grid.get_objects_from_list_of_grid(
            'Debris', neighbours)
        return debris_objects

    def debris_cleaned(self):
        """Find amount of debris cleaned."""
        grid = self.grid
        debris_objects = []
        for obstacle in self.obstacles:
            neighbours = grid.get_neighborhood(
                obstacle.location, obstacle.radius)
            debris_objects += grid.get_objects_from_list_of_grid(
                'Debris', neighbours)
        return list(set(debris_objects))