"""Derived agent class."""

from swarms.lib.agent import Agent
import numpy as np
from swarms.utils.bt import BTConstruct

# from swarms.utils.results import Results
from py_trees import Behaviour, Blackboard
# import copy
from py_trees.meta import inverter
import py_trees
from py_trees.composites import Sequence, Selector
from py_trees.trees import BehaviourTree

from swarms.behaviors.sbehaviors import (
    NeighbourObjects, IsVisitedBefore,
    IsCarrying, IsInPartialAttached, ObjectsOnGrid, IsAgentDead,
    RandomWalk, Move
    )

from swarms.behaviors.scbehaviors import (
    CompositeDrop, CompositeSingleCarry, MoveTowards,
    Explore, CompositeDropPartial, CompositeMultipleCarry, AgentDead, 
    AvoidTrap, ObstacleStuck
)

# import py_trees


class ExploreWithout(Behaviour):
    """Explore behavior for the agents.

    Inherits the Behaviors class from py_trees. This
    behavior combines the privitive behaviors to succesfully explore the
    environment. It combines Randomwalk and Move behaviors.
    """

    def __init__(self, name):
        """Init method for the Explore behavior."""
        super(ExploreWithout, self).__init__(name)
        self.blackboard = Blackboard()
        self.blackboard.shared_content = dict()

    def setup(self, timeout, agent, item=None):
        """Have defined the setup method.

        This method defines the other objects required for the
        behavior. Agent is the actor in the environment,
        item is the name of the item we are trying to find in the
        environment and timeout defines the execution time for the
        behavior.
        """
        self.agent = agent
        self.item = item

        # Check if agent is dead
        adead = inverter(IsAgentDead)('MT_IsAgentDead_0')
        adead.setup(0, self.agent, None)

        # Define the root for the BT
        root = Sequence("Ex_Sequence")

        low = RandomWalk('Ex_RandomWalk')
        low.setup(0, self.agent)

        # Only move if its passable
        passable = ObstacleStuck('MT_Passable_3')
        passable.setup(0, self.agent, 'Obstacles')        

        # Only move if its not deathable 
        # deathable = inverter(IsDeathable)('Ex_Deathable')        
        # deathable.setup(0, self.agent, 'Traps')
        # deathable = AvoidTrap('AvoidTrap')
        # deathable.setup(0, self.agent, 'Traps')

        high = Move('Ex_Move')
        high.setup(0, self.agent)

        chkdead = AgentDead('Adead')
        chkdead.setup(0, self.agent)

        # root.add_children([adead, low, passable, deathable, high, chkdead])
        root.add_children([adead, low, passable, high, chkdead])        

        self.behaviour_tree = BehaviourTree(root)

    def initialise(self):
        """Everytime initialization. Not required for now."""
        pass

    def update(self):
        """Just call the tick method for the sequence.

        This will execute the primitive behaviors defined in the sequence
        """
        self.behaviour_tree.tick()
        return self.behaviour_tree.root.status


class SimForgAgentWithout(Agent):
    """Simulation agent.

    An minimalistic behavior tree for swarm agent
    implementing carry and drop behavior.
    """

    def __init__(self, name, model, xmlstring=None):
        super().__init__(name, model)
        self.location = ()

        self.direction = model.random.rand() * (2 * np.pi)
        self.speed = 2
        self.radius = 3

        self.moveable = True
        self.shared_content = dict()

        self.carryable = False
        # Define a BTContruct object
        self.bt = BTConstruct(None, self)

        class DummyIndividual:
            def __init__(self):
                self.phenotype = None
        dummyind = DummyIndividual()
        self.individual = [dummyind]
        self.individual[0].phenotype = xmlstring

        # self.bt.xmlstring = xmlstring
        # self.bt.construct()
        # 
        # neighobst = NeighbourObjects('NeighbourObjects_Obstacles')
        # neighobst.setup(0, self, 'Obstacles')

        # Drop branch
        dseq = py_trees.composites.Sequence('DSequence')
        iscarrying = IsCarrying('IsCarrying_Food')
        iscarrying.setup(0, self, 'Food')

        neighhub = NeighbourObjects('NeighbourObjects_Hub')
        neighhub.setup(0, self, 'Hub')

        notneighhub = py_trees.meta.inverter(NeighbourObjects)(
            'NeighbourObjects_Hub')
        notneighhub.setup(0, self, 'Hub')

        drop = CompositeDrop('CompositeDrop_Food')
        drop.setup(0, self, 'Food')

        dseq.add_children([neighhub, drop])

        # ## Obstacles and Trap
        # neighobst = ObjectsOnGrid('ObjectsOnGrid_Obstacles')
        # neighobst.setup(0, self, 'Obstacles')

        # neightrap = ObjectsOnGrid('NeighbourObjects_Traps')
        # neightrap.setup(0, self, 'Traps')

        adead = AgentDead('Adead')
        adead.setup(0, self)

        # Carry branch
        cseq = py_trees.composites.Sequence('CSequence')

        neighsite = NeighbourObjects('NeighbourObjects_Sites')
        neighsite.setup(0, self, 'Sites')

        neighfood = NeighbourObjects('NeighbourObjects_Food')
        neighfood.setup(0, self, 'Food')

        invcarrying = py_trees.meta.inverter(IsCarrying)('IsCarrying_Food')
        invcarrying.setup(0, self, 'Food')

        carry = CompositeSingleCarry('CompositeSingleCarry_Food')
        carry.setup(0, self, 'Food')

        cseq.add_children([neighsite, neighfood, invcarrying, carry])

        # Locomotion branch

        # Move to site
        siteseq = py_trees.composites.Sequence('SiteSeq')

        sitefound = IsVisitedBefore('IsVisitedBefore_Sites')
        sitefound.setup(0, self, 'Sites')

        gotosite = MoveTowards('MoveTowards_Sites')
        gotosite.setup(0, self, 'Sites')

        # siteseq.add_children([neighobst, neightrap, sitefound, invcarrying, gotosite])        
        siteseq.add_children([sitefound, invcarrying, gotosite, adead])
        # siteseq.add_children([invcarrying])

        # Move to hub
        hubseq = py_trees.composites.Sequence('HubSeq')

        gotohub = MoveTowards('MoveTowards_Hub')
        gotohub.setup(0, self, 'Hub')

        # hubseq.add_children([neighobst, neightrap, iscarrying, gotohub])
        hubseq.add_children([iscarrying, gotohub, adead])        

        sitenotfound = py_trees.meta.inverter(IsVisitedBefore)(
            'IsVisitedBefore_Sites')
        sitenotfound.setup(0, self, 'Sites')

        explore = ExploreWithout('Explore')
        explore.setup(0, self)

        # randwalk = py_trees.composites.Sequence('Randwalk')
        # randwalk.add_children([neighobst, neightrap, sitenotfound, explore])
        # randwalk.add_children([sitenotfound, explore])        

        locoselect = py_trees.composites.Selector('Move')
        locoselect.add_children([siteseq, hubseq, explore])
        # locoselect.add_children([hubseq, randwalk])

        select = py_trees.composites.Selector('Main')

        select.add_children([dseq, cseq, locoselect])

        self.behaviour_tree = py_trees.trees.BehaviourTree(select)

        # py_trees.display.render_dot_tree(
        #     self.behaviour_tree.root, name=model.pname + '/forgehc')
        # py_trees.logging.level = py_trees.logging.Level.DEBUG
        # py_trees.display.print_ascii_tree(select)

    def step(self):
        # self.bt.behaviour_tree.tick()
        self.behaviour_tree.tick()

    def advance(self):
        pass



class SimForgAgentWith(Agent):
    """Simulation agent.

    An minimalistic behavior tree for swarm agent
    implementing carry and drop behavior.
    """

    def __init__(self, name, model, xmlstring=None):
        super().__init__(name, model)
        self.location = ()

        self.direction = model.random.rand() * (2 * np.pi)
        self.speed = 2
        self.radius = 3

        self.moveable = True
        self.shared_content = dict()

        self.carryable = False
        # Define a BTContruct object
        self.bt = BTConstruct(None, self)

        class DummyIndividual:
            def __init__(self):
                self.phenotype = None
        dummyind = DummyIndividual()
        self.individual = [dummyind]
        self.individual[0].phenotype = xmlstring

        # self.bt.xmlstring = xmlstring
        # self.bt.construct()
        # 
        # neighobst = NeighbourObjects('NeighbourObjects_Obstacles')
        # neighobst.setup(0, self, 'Obstacles')

        # Drop branch
        dseq = py_trees.composites.Sequence('DSequence')
        iscarrying = IsCarrying('IsCarrying_Food')
        iscarrying.setup(0, self, 'Food')

        neighhub = NeighbourObjects('NeighbourObjects_Hub')
        neighhub.setup(0, self, 'Hub')

        notneighhub = py_trees.meta.inverter(NeighbourObjects)(
            'NeighbourObjects_Hub')
        notneighhub.setup(0, self, 'Hub')

        drop = CompositeDrop('CompositeDrop_Food')
        drop.setup(0, self, 'Food')

        dseq.add_children([neighhub, drop])

        # ## Obstacles and Trap
        # neighobst = ObjectsOnGrid('ObjectsOnGrid_Obstacles')
        # neighobst.setup(0, self, 'Obstacles')

        # neightrap = ObjectsOnGrid('NeighbourObjects_Traps')
        # neightrap.setup(0, self, 'Traps')

        adead = AgentDead('Adead')
        adead.setup(0, self)

        # Carry branch
        cseq = py_trees.composites.Sequence('CSequence')

        neighsite = NeighbourObjects('NeighbourObjects_Sites')
        neighsite.setup(0, self, 'Sites')

        neighfood = NeighbourObjects('NeighbourObjects_Food')
        neighfood.setup(0, self, 'Food')

        invcarrying = py_trees.meta.inverter(IsCarrying)('IsCarrying_Food')
        invcarrying.setup(0, self, 'Food')

        carry = CompositeSingleCarry('CompositeSingleCarry_Food')
        carry.setup(0, self, 'Food')

        cseq.add_children([neighsite, neighfood, invcarrying, carry])

        # Locomotion branch

        # Move to site
        siteseq = py_trees.composites.Sequence('SiteSeq')

        sitefound = IsVisitedBefore('IsVisitedBefore_Sites')
        sitefound.setup(0, self, 'Sites')

        gotosite = MoveTowards('MoveTowards_Sites')
        gotosite.setup(0, self, 'Sites')

        # siteseq.add_children([neighobst, neightrap, sitefound, invcarrying, gotosite])        
        siteseq.add_children([sitefound, invcarrying, gotosite, adead])
        # siteseq.add_children([invcarrying])

        # Move to hub
        hubseq = py_trees.composites.Sequence('HubSeq')

        gotohub = MoveTowards('MoveTowards_Hub')
        gotohub.setup(0, self, 'Hub')

        # hubseq.add_children([neighobst, neightrap, iscarrying, gotohub])
        hubseq.add_children([iscarrying, gotohub, adead])        

        sitenotfound = py_trees.meta.inverter(IsVisitedBefore)(
            'IsVisitedBefore_Sites')
        sitenotfound.setup(0, self, 'Sites')

        explore = Explore('Explore')
        explore.setup(0, self)

        # randwalk = py_trees.composites.Sequence('Randwalk')
        # randwalk.add_children([neighobst, neightrap, sitenotfound, explore])
        # randwalk.add_children([sitenotfound, avoidt, explore])        

        locoselect = py_trees.composites.Selector('Move')
        locoselect.add_children([siteseq, hubseq, explore])
        # locoselect.add_children([hubseq, randwalk])

        select = py_trees.composites.Selector('Main')

        select.add_children([dseq, cseq, locoselect])

        self.behaviour_tree = py_trees.trees.BehaviourTree(select)

        # py_trees.display.render_dot_tree(
        #     self.behaviour_tree.root, name=model.pname + '/forgehc')
        # py_trees.logging.level = py_trees.logging.Level.DEBUG
        # py_trees.display.print_ascii_tree(select)

    def step(self):
        # self.bt.behaviour_tree.tick()
        self.behaviour_tree.tick()

    def advance(self):
        pass


class EvolAgent(Agent):
    """An minimalistic swarm agent."""

    def __init__(self, name, model):
        """Initialize the agent."""
        super().__init__(name, model)
        self.location = ()

        self.direction = model.random.rand() * (2 * np.pi)
        self.speed = 2
        self.radius = 3
        self.results = "file"  # This can take 2 values. db or file

        # self.exchange_time = model.random.randint(2, 4)
        # This doesn't help. Maybe only perform genetic operations when
        # an agents meet 10% of its total population
        # """
        self.operation_threshold = 2
        self.genome_storage = []

        # Define a BTContruct object
        self.bt = BTConstruct(None, self)

        # self.blackboard = Blackboard()
        # self.blackboard.shared_content = dict()

        self.shared_content = dict()
        # self.shared_content = dict(
        self.carryable = False
        self.beta = 0.0001
        self.food_collected = 0
        # Grammatical Evolution part
        from ponyge.algorithm.parameters import Parameters
        parameter = Parameters()
        parameter_list = ['--parameters', '../..,' + model.parm]
        # Comment when different results is desired.
        # Else set this for testing purpose
        # parameter.params['RANDOM_SEED'] = name
        # # np.random.randint(1, 99999999)
        parameter.params['POPULATION_SIZE'] = self.operation_threshold // 2
        parameter.set_params(parameter_list)
        self.parameter = parameter
        individual = initialisation(self.parameter, 1)
        individual = evaluate_fitness(individual, self.parameter)

        self.individual = individual
        self.bt.xmlstring = self.individual[0].phenotype
        self.bt.construct()

        self.diversity_fitness = self.individual[0].fitness
        self.delayed_reward = 0
        # Location history
        self.location_history = set()
        self.timestamp = 0
        self.step_count = 0

        self.fitness_name = True

    def get_food_in_hub(self):
        """Return food in the hub."""
        grid = self.model.grid
        hub_loc = self.model.hub.location
        neighbours = grid.get_neighborhood(hub_loc, 10)
        food_objects = grid.get_objects_from_list_of_grid('Food', neighbours)
        agent_food_objects = []
        for food in food_objects:
            if (
                food.agent_name == self.name and
                    food.phenotype == self.individual[0].phenotype):
                agent_food_objects.append(food)
        return agent_food_objects

    def detect_food_carrying(self):
        """Detect if the agent is carrying food."""
        if len(self.attached_objects) > 0:
            print('Food carying', self.name, self.attached_objects)
            output = py_trees.display.ascii_tree(self.bt.behaviour_tree.root)
            print(output)

    def store_genome(self, cellmates):
        """Store the genome from neighbours."""
        # cellmates.remove(self)
        # self.genome_storage += [agent.individual[0] for agent in cellmates]
        for agent in cellmates:
            if agent.food_collected > 0:
                self.genome_storage += agent.individual
            elif len(agent.attached_objects) > 0:
                self.genome_storage += agent.individual
            elif agent.exploration_fitness() > 10:
                self.genome_storage += agent.individual

    def exchange_chromosome(self,):
        """Perform genetic operations."""
        # print('from exchange', self.name)
        individuals = self.genome_storage
        parents = selection(self.parameter, individuals)
        cross_pop = crossover(self.parameter, parents)
        new_pop = mutation(self.parameter, cross_pop)
        new_pop = evaluate_fitness(new_pop, self.parameter)
        individuals = replacement(self.parameter, new_pop, individuals)
        individuals.sort(reverse=False)
        self.individual = [individuals[0]]
        self.individual[0].fitness = 0
        self.genome_storage = []

    def genetic_step(self):
        """Additional procedures called after genecti step."""
        self.delayed_reward = self.individual[0].fitness
        self.exchange_chromosome()
        self.bt.xmlstring = self.individual[0].phenotype
        self.bt.construct()
        self.food_collected = 0
        self.location_history = set()
        self.timestamp = 0
        self.diversity_fitness = self.individual[0].fitness

    def overall_fitness(self):
        """Compute complete fitness.

        Goals are represented by objective function. We use combination of
        objective function to define overall fitness of the agents
        performance.
        """
        # Use a decyaing function to generate fitness
        # Use two step decaying function
        # First block gives importance to exploration and when as soon
        # food has been found, the next block will focus on dropping
        # the food on hub
        self.individual[0].fitness = (1 - self.beta) * self.delayed_reward \
            + self.exploration_fitness() + self.carrying_fitness() \
            + self.food_collected

    def carrying_fitness(self):
        """Compute carrying fitness.

        This fitness supports the carrying behavior of
        the agents.
        """
        return len(self.attached_objects) * (self.timestamp)

    def exploration_fitness(self):
        """Compute the exploration fitness."""
        # Use exploration space as fitness values
        return len(self.location_history) - 1

    # New Agent methods for behavior based robotics
    def sense(self):
        """Sense included in behavior tree."""
        pass

    def plan(self):
        """Plan not required for now."""
        pass

    def step(self):
        """Agent action at a single time step."""
        # py_trees.logging.level = py_trees.logging.Level.DEBUG
        # output = py_trees.display.ascii_tree(self.bt.behaviour_tree.root)

        # Couting variables
        self.timestamp += 1
        self.step_count += 1

        # Increase beta
        self.beta = self.step_count / self.model.iter

        self.location_history.add(self.location)

        # Compute the behavior tree
        self.bt.behaviour_tree.tick()

        # Find the no.of food collected from the BT execution
        self.food_collected = len(self.get_food_in_hub())

        # Computes overall fitness using Beta function
        self.overall_fitness()

        cellmates = self.model.grid.get_objects_from_grid(
            type(self).__name__, self.location)

        # Create a results instance and save it to a file
        """
        self.results = Results(
            self.model.pname, self.model.connect, self.model.sn, self.name,
            self.step_count, self.timestamp, self.beta,
            self.individual[0].fitness,
            self.diversity_fitness, self.exploration_fitness(),
            self.food_collected, len(cellmates), self.individual[0].genome,
            self.individual[0].phenotype, self.bt
            )
        """
        # Save the results to a db
        # self.results.save_to_file()

        # Logic for gentic operations.
        # If the genome storage has enough genomes and agents has done some
        # exploration then compute the genetic step OR
        # 600 time step has passed and the agent has not done anything useful
        # then also perform genetic step
        storage_threshold = len(
            self.genome_storage) >= (self.model.num_agents / 1.4)
        if storage_threshold:
            self.genetic_step()
        elif (
            storage_threshold is False and self.timestamp > 50 and
                self.exploration_fitness() < 10):
            individual = initialisation(self.parameter, 10)
            individual = evaluate_fitness(individual, self.parameter)
            self.genome_storage = individual
            self.genetic_step()

        # If neighbours found, store the genome
        if len(cellmates) > 1:
            self.store_genome(cellmates)

    def advance(self):
        """Require for staged activation."""
        pass
