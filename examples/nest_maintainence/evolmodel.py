"""Inherited model class."""

from swarms.lib.model import Model
from swarms.lib.time import SimultaneousActivation
from swarms.lib.space import Grid
from swarms.utils.jsonhandler import JsonData
from swarms.utils.results import Best, Experiment
from swarms.utils.db import Connect
from evolagent import EvolAgent     # noqa: F401
from swarms.lib.objects import Hub, Sites, Debris, Obstacles  # noqa: F401
import os
import imp
import datetime
import numpy as np

filename = os.path.join(imp.find_module("swarms")[1] + "/utils/world.json")


class EvolModel(Model):
    """A environemnt to model swarms."""

    def __init__(
            self, N, width, height, grid=10, iter=100000, seed=None,
            expname='NestM', agent='EvolAgent', parm='nm.txt', fitid=0):
        """Initialize the attributes."""
        if seed is None:
            super(EvolModel, self).__init__(seed=None)
        else:
            super(EvolModel, self).__init__(seed)

        self.runid = datetime.datetime.now().strftime(
            "%s") + str(self.random.randint(1, 10000, 1)[0])

        self.pname = '/'.join(
            os.getcwd().split('/')[:-2]) + '/results/' \
            + str(fitid) + '/' \
            + self.runid + '-' + str(iter) + expname

        self.stepcnt = 1
        self.iter = iter
        self.top = None
        # Create db connection
        connect = Connect('swarm', 'swarm', 'swarm', 'localhost')
        self.connect = connect.tns_connect()

        # Fill out the experiment table
        self.experiment = Experiment(
            self.connect, self.runid, N, seed, expname + '-'+ str(fitid),
            iter, width, height, grid)
        self.experiment.insert_experiment()

        self.sn = self.experiment.sn

        # Create a folder to store results
        os.mkdir(self.pname)

        self.num_agents = N

        self.grid = Grid(width, height, grid)

        self.schedule = SimultaneousActivation(self)

        self.agents = []
        self.parm = parm

        self.modes = {
            0: (True, False, False, False),
            1: (True, True, False, False),
            2: (True, True, True, False),
            3: (True, True, True, True)
        }
        self.fitmode = fitid

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
            for i in range(4):
                dx, dy = self.random.randint(5, 10, 2)
                dx = self.hub.location[0] + 25 + dx
                dy = self.hub.location[1] + 25 + dy
                o = Obstacles(id=i, location=(dx, dy), radius=10)
                self.grid.add_object_to_grid(o.location, o)
                self.obstacles.append(o)
        except AttributeError:
            pass

        # This doesn't change so no need to compute everytime
        grid = self.grid
        hub_loc = self.hub.location

        neighbours_in = grid.get_neighborhood(hub_loc, 25)
        neighbours_out = grid.get_neighborhood(hub_loc, 50)
        self.neighbours = list(set(neighbours_out) - set(neighbours_in))

    def step(self):
        """Step through the environment."""
        # Gather info from all the agents
        # self.top = self.gather_info()
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
            foraging[id] = self.agents[id].debris_collected
            fittest[id] = self.agents[id].individual[0].fitness
        beta = self.agents[-1].beta
        """
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
        """
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
        ffitness = self.agents[idx].debris_collected
        efitness = self.agents[idx].exploration_fitness()
        phenotype = self.agents[idx].individual[0].phenotype

        best_agent = Best(
            self.pname, self.connect, self.sn, idx, header, self.stepcnt, beta,
            ofitness, dfitness, efitness, ffitness, 'None'
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

    def find_higest_debris_collector(self):
        """Find the best agent to collect debris."""
        fitness = self.agents[0].debris_collected
        fittest = self.agents[0]
        for agent in self.agents:
            if agent.debris_collected > fitness:
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

    def nestm_percent(self):
        """Compute the percent of the total debris cleared from the hub."""
        grid = self.grid
        debris_objects = []
        for obstacle in self.obstacles:
            neighbours = grid.get_neighborhood(
                obstacle.location, obstacle.radius)
            debris_objects += grid.get_objects_from_list_of_grid(
                'Debris', neighbours)

        total_debry_weights = sum([debry.weight for debry in self.debris])
        total_moved_debry = sum([debry.weight for debry in debris_objects])
        return ((total_moved_debry * 1.0) / total_debry_weights) * 100