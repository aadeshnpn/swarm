"""Inherited model class."""

import psycopg2
from swarms.lib.model import Model
from swarms.lib.time import SimultaneousActivation
# RandomActivation, StagedActivation
from swarms.lib.space import Grid
from swarms.utils.jsonhandler import JsonData
from swarms.utils.results import Best, Experiment
from swarms.utils.db import Connect
from swarms.utils.ui import UI
from agent import LearningAgent, ExecutingAgent  # noqa : F041
from swarms.lib.objects import (    # noqa : F401
    Hub, Sites, Food, Debris, Obstacles, Traps)
import os
from pathlib import Path
# import imp
import datetime
import numpy as np
from flloat.parser.ltlf import LTLfParser


# filename = os.path.join(imp.find_module("swarms")[1] + "/utils/world.json")
projectdir = Path(__file__).resolve().parent
# projectdir = '/home/aadeshnpn/Documents/BYU/HCMI/resilience/swarm/examples/resilience/'
filename = os.path.join(projectdir, 'world.json')
# projectdir = "/home/aadeshnpn/Documents/BYU/HCMI/resilience/swarm/examples"
# filename = os.path.join(projectdir + "/resilience_evolution/world.json")



class ForagingModel(Model):
    """A environemnt to model foraging environment."""

    def __init__(
            self, N, width, height, grid=10, iter=100000,
            seed=None, name='SForagingPPA1', viewer=False, parent=None, ratio=1.0, db=False):
        """Initialize the attributes."""
        if seed is None:
            super(ForagingModel, self).__init__(seed=None)
        else:
            super(ForagingModel, self).__init__(seed)

        # Create a unique experiment id
        self.runid = datetime.datetime.now().strftime(
            "%s") + str(self.random.randint(1, 10000, 1)[0])

        # Create the experiment folder
        # If parent folder exits create inside it
        if parent is not None:
            self.pname = parent + '/' + str(self.runid) + '_' + str(ratio) +'_' +name
            # Path(self.pname).mkdir(parents=True, exist_ok=True)
        else:
            self.pname = os.path.join(
                '/tmp', 'swarm', 'data', 'experiments',
                str(N), str(iter), str(self.runid) + name
                )
        Path(self.pname).mkdir(parents=True, exist_ok=True)

        # Define some parameters to count the step
        self.stepcnt = 1
        self.iter = iter

        # UI
        self.viewer = viewer

        # Create db connection
        if db:
            try:
                connect = Connect('swarm', 'swarm', 'swarm', 'localhost')
                self.connect = connect.tns_connect()
            except psycopg2.OperationalError:
                self.connect = None
        else:
            self.connect = None
        # Fill out the experiment table
        self.experiment = Experiment(
            self.connect, self.runid, N, seed, name,
            iter, width, height, grid, db=db)
        if db:
            self.experiment.insert_experiment()
        # Get the primary key of the experiment table for future use
        self.sn = self.experiment.sn

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
        # self.traps = self.render.objects['traps'][0]
        # self.obstacles = self.render.objects['obstacles'][0]
        self.total_food_units = 0
        self.foods = []
        try:
            self.site = self.render.objects['sites'][0]
            for i in range(self.num_agents * 1):
                f = Food(
                    i, location=self.site.location, radius=self.site.radius)
                f.agent_name = None
                self.grid.add_object_to_grid(f.location, f)
                self.total_food_units += f.weight
                f.phenotype = dict()
                self.foods.append(f)
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
        # diversity = np.ones(len(self.agents))
        exploration = np.ones(len(self.agents))
        foraging = np.ones(len(self.agents))
        fittest = np.ones(len(self.agents))
        diversity = np.ones(len(self.agents))
        postcondition = np.ones(len(self.agents))
        constraints = np.ones(len(self.agents))
        selector = np.ones(len(self.agents))
        for id in range(len(self.agents)):
            exploration[id] = self.agents[id].exploration_fitness()
            foraging[id] = self.agents[id].food_collected
            fittest[id] = self.agents[id].individual[0].fitness
            diversity[id] = self.agents[id].diversity_fitness
            postcondition[id] = self.agents[id].postcond_reward
            constraints[id] = self.agents[id].constraints_reward
            selector[id] = self.agents[id].selectors_reward

        beta = self.agents[-1].beta

        mean = Best(
            self.pname, self.connect, self.sn, 1, 'MEAN', self.stepcnt,
            beta, np.mean(fittest), np.mean(diversity), np.mean(exploration),
            np.mean(foraging), np.mean(postcondition), np.mean(constraints),
            np.mean(selector), "None", "None", db=False
            )
        mean.save()

        std = Best(
            self.pname, self.connect, self.sn, 1, 'STD', self.stepcnt, beta,
            np.std(fittest), np.std(diversity), np.std(exploration),
            np.std(foraging), np.mean(postcondition), np.mean(constraints),
            np.mean(selector), "None", "None", db=False
            )
        std.save()

        # Compute best agent for each fitness
        self.best_agents(diversity, beta, "DIVERSE")
        self.best_agents(exploration, beta, "EXPLORE")
        self.best_agents(foraging, beta, "FORGE")
        self.best_agents(postcondition, beta, "PCOND")
        self.best_agents(constraints, beta, "CNSTR")
        self.best_agents(selector, beta, "SELECT")
        self.best_agents(fittest, beta, "OVERALL")
        return np.argmax(foraging)

    def best_agents(self, data, beta, header):
        """Find the best agents in each category."""
        idx = np.argmax(data)
        # dfitness = self.agents[idx].diversity_fitness
        ofitness = self.agents[idx].individual[0].fitness
        ffitness = self.agents[idx].food_collected
        efitness = self.agents[idx].exploration_fitness()
        pfitness = self.agents[idx].diversity_fitness
        pcfitness = self.agents[idx].postcond_reward
        sefitness = self.agents[idx].selectors_reward
        cnstrfitness = self.agents[idx].constraints_reward
        phenotype = "None" # self.agents[idx].individual[0].phenotype

        best_agent = Best(
            self.pname, self.connect, self.sn, idx, header, self.stepcnt, beta,
            ofitness, pfitness, efitness, ffitness, pcfitness, cnstrfitness,
            sefitness, phenotype, "None", db=False
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
        neighbours = grid.get_neighborhood(hub_loc, self.hub.radius)
        food_objects = grid.get_objects_from_list_of_grid('Food', neighbours)
        _, hub_grid = grid.find_grid(hub_loc)
        for food in self.foods:
            _, food_grid = grid.find_grid(food.location)
            if food_grid == hub_grid:
                food_objects += [food]
        food_objects = set(food_objects)
        total_food_weights = sum([food.weight for food in food_objects])
        return np.round(((total_food_weights * 1.0) / self.total_food_units) * 100, 1)

    def no_agent_dead(self):
        # grid = self.grid
        # trap_loc = self.traps.location
        # neighbours = grid.get_neighborhood(trap_loc, 10)
        # agents = grid.get_objects_from_list_of_grid(type(self.agents[0]).__name__, neighbours)
        # return sum([1 if a.dead else 0 for a in agents])
        return 0


class EvolveModel(ForagingModel):
    """A environemnt to model swarms."""

    def __init__(
            self, N, width, height, grid=10, iter=100000,
            seed=None, name="EvoSForgeNewPPA1", viewer=False, db=False):
        """Initialize the attributes."""
        super(EvolveModel, self).__init__(
            N, width, height, grid, iter, seed, name, viewer, db=db)
        self.parser = LTLfParser()

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

            if random_init:
                # Add the agent to a random grid cell
                x = self.random.randint(
                    -self.grid.width / 2, self.grid.width / 2)
                y = self.random.randint(
                    -self.grid.height / 2, self.grid.height / 2)
            else:
                try:
                    x, y = self.hub.location
                except AttributeError:
                    x, y = 0, 0
            a.location = (x, y)
            self.grid.add_object_to_grid((x, y), a)
            # a.operation_threshold = 2  # self.num_agents // 10
            self.agents.append(a)

    def behavior_sampling(self, method='ratio', ratio_value=0.2, phenotype=None):
        """Extract phenotype of the learning agents.

        Sort the agents based on the overall fitness and then based on the
        method extract phenotype of the agents.
        Method can take {'ratio','higest','sample'}
        """
        # sorted_agents = sorted(
        #    self.agents, key=lambda x: x.individual[0].fitness, reverse=True)
        if phenotype is None:
            phenotypes = dict()
            # Get the phenotypes collected from the agent
            for agent in self.agents:
                phenotypes = {**agent.phenotypes, **phenotypes}
            # Sort the phenotypes
            phenotypes, _ = zip(
                *sorted(phenotypes.items(), key=lambda x: (
                    x), reverse=True))
        else:
            phenotypes = phenotype
        # Just for testing. Not needed

        if method == 'ratio':
            upper_bound = ratio_value * self.num_agents
            # selected_agents = self.agents[0:int(upper_bound)]
            # selected_phenotype = [
            #    agent.individual[0].phenotype for agent in selected_agents]
            selected_phenotype = list(phenotypes)[:int(upper_bound)]
            return selected_phenotype
        else:
            # return [sorted_agents[0].individual[0].phenotype]
            return [phenotypes[0]]

    def behavior_sampling_objects(self, method='ratio', ratio_value=0.2, phenotype=None):
        """Extract phenotype of the learning agents based on the objects.

        Sort the phenotye based on the overall fitness and then based on the
        method extract phenotype of the agents.
        Method can take {'ratio','higest','sample'}
        """
        # sorted_agents = sorted(
        #    self.agents, key=lambda x: x.individual[0].fitness, reverse=True)
        # phenotypes = dict()
        # Get the phenotypes collected from the agent
        if phenotype is None:
            phenotypes = self.phenotype_attached_objects()

            for agent in self.agents:
                phenotypes = {**agent.phenotypes, **phenotypes}

            print('total phenotype from behavior sampling', len(phenotypes))
            # Sort the phenotypes
            phenotypes, _ = zip(
                *sorted(phenotypes.items(), key=lambda x: (
                    x[1]), reverse=True))
        else:
            phenotypes = phenotype

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

    def step(self):
        """Step through the environment."""
        # Gather info to plot the graph
        try:
            # self.gather_info()
            pass
            # agent = self.agents[idx]
            # print(
            #    idx, agent.individual[0].phenotype,
            #    agent.individual[0].fitness, agent.food_collected)
        except FloatingPointError:
            pass

        # Next step
        self.schedule.step()
        # input('Enter to continue' + str(self.stepcnt))
        # Increment the step count
        self.stepcnt += 1


class ValidationModel(ForagingModel):
    """A environemnt to validate swarm behaviors."""

    def __init__(
            self, N, width, height, grid=10, iter=100000,
            seed=None, name="ValidateSForgeNewPPA1", viewer=False,
            parent=None, ratio=1.0, db=False):
        """Initialize the attributes."""
        super(ValidationModel, self).__init__(
            N, width, height, grid, iter, seed, name, viewer, parent, ratio, db)

    def create_agents(self, random_init=False, phenotypes=None):
        """Initialize agents in the environment."""
        # Variable to tell how many agents will have the same phenotype
        # bound = np.ceil((self.num_agents * 1.0) / len(phenotypes))
        j = 0
        # Create agents
        for i in range(self.num_agents):
            # print (i, j, self.xmlstrings[j])
            a = ExecutingAgent(i, self, xmlstring=phenotypes)
            # a = TestingAgent(i, self, xmlstring=phenotypes[j])
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
            else:
                try:
                    x, y = self.hub.location
                except AttributeError:
                    x, y = 0, 0

            a.location = (x, y)
            self.grid.add_object_to_grid((x, y), a)
            a.operation_threshold = 2  # self.num_agents // 10
            self.agents.append(a)
            j += 1
            if j >= len(phenotypes):
                j = 0

    def behavior_sampling(self, method='ratio', ratio_value=0.2, phenotype=None):
        """Extract phenotype of the learning agents.

        Sort the agents based on the overall fitness and then based on the
        method extract phenotype of the agents.
        Method can take {'ratio','higest','sample'}
        """
        # sorted_agents = sorted(
        #    self.agents, key=lambda x: x.individual[0].fitness, reverse=True)
        if phenotype is None:
            phenotypes = dict()
            # Get the phenotypes collected from the agent
            for agent in self.agents:
                phenotypes = {**agent.phenotypes, **phenotypes}
            # Sort the phenotypes
            phenotypes, _ = zip(
                *sorted(phenotypes.items(), key=lambda x: (
                    x), reverse=True))
        else:
            phenotypes = phenotype
        # Just for testing. Not needed

        if method == 'ratio':
            upper_bound = round(ratio_value * len(phenotypes))
            selected_phenotype = list(phenotypes)[:int(upper_bound)]
            return selected_phenotype
        else:
            # return [sorted_agents[0].individual[0].phenotype]
            return [phenotypes[0]]


class TestModel(ForagingModel):
    """A environemnt to test swarm behavior performance."""

    def __init__(
            self, N, width, height, grid=10, iter=100000,
            seed=None, name="TestSForgeNewPPA1", viewer=False,
            parent=None, ratio=1.0, db=False):
        """Initialize the attributes."""
        super(TestModel, self).__init__(
            N, width, height, grid, iter, seed, name, viewer, parent, ratio, db)

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
            else:
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

    def behavior_sampling(self, method='ratio', ratio_value=0.2, phenotype=None):
        """Extract phenotype of the learning agents.

        Sort the agents based on the overall fitness and then based on the
        method extract phenotype of the agents.
        Method can take {'ratio','higest','sample'}
        """
        # sorted_agents = sorted(
        #    self.agents, key=lambda x: x.individual[0].fitness, reverse=True)
        if phenotype is None:
            phenotypes = dict()
            # Get the phenotypes collected from the agent
            for agent in self.agents:
                phenotypes = {**agent.phenotypes, **phenotypes}
            # Sort the phenotypes
            phenotypes, _ = zip(
                *sorted(phenotypes.items(), key=lambda x: (
                    x), reverse=True))
        else:
            phenotypes = phenotype
        # Just for testing. Not needed

        if method == 'ratio':
            upper_bound = round(ratio_value * len(phenotypes))
            # selected_agents = self.agents[0:int(upper_bound)]
            # selected_phenotype = [
            #    agent.individual[0].phenotype for agent in selected_agents]
            selected_phenotype = list(phenotypes)[:int(upper_bound)]
            return selected_phenotype
        else:
            # return [sorted_agents[0].individual[0].phenotype]
            return [phenotypes[0]]


class ViewerModel(ForagingModel):
    """A environemnt to test swarm behavior performance."""

    def __init__(
            self, N, width, height, grid=10, iter=100000,
            seed=None, name="ViewerSForgeNewPPA1", viewer=True, db=False):
        """Initialize the attributes."""
        super(ViewerModel, self).__init__(
            N, width, height, grid, iter, seed, name, viewer, db=db)

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
            else:
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

        if self.viewer:
            self.ui = UI(
                (self.grid.width, self.grid.height), [self.hub], self.agents,
                sites=[self.site], food=self.foods, obstacles=[self.obstacles], traps=[self.traps])

    def step(self):
        """Step through the environment."""
        # Next step
        self.schedule.step()

        # Increment the step count
        self.stepcnt += 1

        # If viewer required do take a step in UI
        if self.viewer:
            self.ui.step()


class SimForgModel(Model):
    """A environemnt to model swarms."""

    def __init__(
            self, N, width, height, grid=10, iter=100000,
            xmlstrings=None, seed=None, viewer=False, pname=None,
            agent=ExecutingAgent, expsite=None, trap=5, obs=5, notrap=1, noobs=1):
        """Initialize the attributes."""
        if seed is None:
            super(SimForgModel, self).__init__(seed=None)
        else:
            super(SimForgModel, self).__init__(seed)

        self.width = width
        self.height = height
        self.stepcnt = 1
        self.iter = iter
        self.xmlstrings = xmlstrings

        self.viewer = viewer
        self.agent = agent
        self.expsite = expsite
        self.trap_radius = trap
        self.obs_radius = obs
        self.no_trap = notrap
        self.no_obs = noobs
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
        while True:
            self.runid = datetime.datetime.now().strftime(
                    "%s") + str(self.random.randint(1, 10000, 1)[0])
            if pname is None:
                self.pname = os.getcwd() + '/' + self.runid + "SForagingSimulation"
            else:
                self.pname = pname + '/' + self.runid + "SForagingSimulation"
            if not Path(self.pname).exists():
                Path(self.pname).mkdir(parents=True, exist_ok=False)
                break

        self.num_agents = N

        self.grid = Grid(width, height, grid)

        self.schedule = SimultaneousActivation(self)
        self.traps = []
        self.obstacles = []

        self.agents = []

        bound = np.ceil((self.num_agents * 1.0) / len(self.xmlstrings))

        j = 0
        # Create agents
        for i in range(self.num_agents):
            # print (i, j, self.xmlstrings[j])
            a = self.agent(i, self, xmlstring=self.xmlstrings)
            self.schedule.add(a)
            # Add the hub to agents memory
            # Add the hub to agents memory
            # a.shared_content['Hub'] = {self.hub}
            # Initialize the BT. Since the agents are normal agents just
            # use the phenotype
            a.construct_bt()
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

    def place_site(self):
        theta = np.linspace(0, 2*np.pi, 36)
        while True:
            t = self.random.choice(theta, 1, replace=False)[0]
            x = int(self.hub.location[0] + np.cos(t) * self.expsite)
            y = int(self.hub.location[0] + np.sin(t) * self.expsite)
            location = (x, y)
            radius = 10
            q_value = 0.9
            other_bojects = self.grid.get_objects_from_list_of_grid(None, self.grid.get_neighborhood((x,y), radius))
            if len(other_bojects) == 0:
                self.site = Sites(
                        0, location, radius, q_value=q_value)
                self.grid.add_object_to_grid(location, self.site)
                break

    def place_static_objs(self, obj, radius):
        theta = np.linspace(0, 2*np.pi, 36)
        while True:
            dist = self.random.choice(range(25, self.width//2, 5))
            t = self.random.choice(theta, 1, replace=False)[0]
            x = int(0 + np.cos(t) * dist)
            y = int(0 + np.sin(t) * dist)
            location = (x, y)
            other_bojects = self.grid.get_objects_from_list_of_grid(None, self.grid.get_neighborhood((x,y), radius))
            # print(obj, radius, location)
            if len(other_bojects) == 0:
                envobj = obj(
                        dist, location, radius)
                self.grid.add_object_to_grid(location, envobj)
                # bojects = self.grid.get_objects_from_list_of_grid('Traps', self.grid.get_neighborhood(location, radius))
                # print('reverse', bojects)
                if isinstance(envobj, Traps):
                    self.traps += [envobj]
                if isinstance(envobj, Obstacles):
                    self.obstacles += [envobj]
                break

    def create_environment_object(self, jsondata, obj):
        """Create env from jsondata."""
        name = obj.__name__.lower()
        temp_list = []
        i = 0
        for json_object in jsondata[name]:
            location = (json_object["x"], json_object["y"])
            if "q_value" in json_object:
                temp_obj = None
                pass
                # temp_obj = obj(
                #     i, location, json_object["radius"], q_value=json_object[
                #         "q_value"])
            else:
                if name == 'traps':
                    # temp_obj = obj(i, location, self.trap_radius)
                    temp_obj = None
                    for t in range(self.no_trap):
                        self.place_static_objs(Traps, self.trap_radius)
                elif name =='obstacles':
                    temp_obj = None
                    for o in range(self.no_obs):
                        self.place_static_objs(Obstacles, self.obs_radius)
                    # temp_obj = obj(i, location, self.obs_radius)
                elif name == 'hub':
                    temp_obj = obj(i, location, json_object["radius"])
            if temp_obj is not None:
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
        # self.obstacles = self.render.objects['obstacles'][0]
        # print(self.obstacles.passable)
        # self.traps = self.render.objects['traps'][0]
        # add site with random distances
        self.place_site()
        # print(self.traps, self.obstacles)
        # print(self.traps, self.obstacles, self.hub, self.site)
        # location = (self.expsite["x"], self.expsite["y"])
        # self.site = Sites(
        #         0, location, self.expsite["radius"], q_value=self.expsite[
        #             "q_value"])

        # self.grid.add_object_to_grid(location, self.site)

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
        # grid = self.grid
        # food_loc = self.hub.location
        # neighbours = grid.get_neighborhood(food_loc, self.hub.radius)
        # food_objects = grid.get_objects_from_list_of_grid('Food', neighbours)
        # return len(food_objects)
        grid = self.grid
        hub_loc = self.hub.location
        neighbours = grid.get_neighborhood(hub_loc, self.hub.radius)
        food_objects = grid.get_objects_from_list_of_grid('Food', neighbours)
        _, hub_grid = grid.find_grid(hub_loc)
        for food in self.foods:
            _, food_grid = grid.find_grid(food.location)
            if food_grid == hub_grid:
                food_objects += [food]
        food_objects = set(food_objects)
        # total_food_weights = sum([food.weight for food in food_objects])
        # return np.round(((total_food_weights * 1.0) / self.total_food_units) * 100, 1)
        return len(food_objects)

    def food_in_loc(self, loc):
        """Find amount of food in hub."""
        grid = self.grid
        neighbours = grid.get_neighborhood(loc, 10)
        food_objects = grid.get_objects_from_list_of_grid('Food', neighbours)
        return food_objects

    def no_agent_dead(self):
        grid = self.grid
        no_dead = 0
        for trap in self.traps:
            trap_loc = trap.location
            neighbours = grid.get_neighborhood(trap_loc, 10)
            agents = grid.get_objects_from_list_of_grid(type(self.agents[0]).__name__, neighbours)
            no_dead += sum([1 if a.dead else 0 for a in agents])
        return no_dead
