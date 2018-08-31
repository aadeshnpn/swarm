"""Experiment script to run Single source foraging simulation."""

from model import EnvironmentModel, RunEnvironmentModel
# from swarms.utils.jsonhandler import JsonData
from swarms.utils.graph import Graph, GraphACC
from joblib import Parallel, delayed
from swarms.utils.results import SimulationResults

# Global variables for width and height
width = 100
height = 100

UI = False


def extract_phenotype(agents, method='ratio'):
    """Extract phenotype of the learning agents.

    Sort the agents based on the overall fitness and then based on the
    method extract phenotype of the agents.
    Method can take {'ratio','higest','sample'}
    """
    sorted_agents = sorted(
        agents, key=lambda x: x.individual[0].fitness, reverse=True)

    if method == 'ratio':
        ratio_value = 0.4
        upper_bound = ratio_value * len(agents)
        selected_agents = agents[0:int(upper_bound)]
        selected_phenotype = [
            agent.individual[0].phenotype for agent in selected_agents]
        return selected_phenotype
    else:
        return [sorted_agents[0].individual[0].phenotype]


def simulate(agents, iteration):
    """Test the performane of evolved behavior."""
    # phenotype = agent.individual[0].phenotype
    # phenotypes = extract_phenotype(agents)
    phenotypes = agents
    # iteration = 10000
    threshold = 1.0
    sim = RunEnvironmentModel(
        100, 100, 100, 10, iter=iteration, xmlstrings=phenotypes)
    sim.build_environment_from_json()

    # for all agents store the information about hub
    # Also
    for agent in sim.agents:
        agent.shared_content['Hub'] = {sim.hub}
        agent.shared_content['Sites'] = {sim.site}

    simresults = SimulationResults(
        sim.pname, sim.connect, sim.sn, sim.stepcnt, sim.food_in_hub(),
        phenotypes[0]
        )
    simresults.save_phenotype()
    simresults.save_to_file()

    # Iterate and execute each step in the environment
    for i in range(iteration):
        # For every iteration we need to store the results
        # Save them into db or a file
        sim.step()
        simresults = SimulationResults(
            sim.pname, sim.connect, sim.sn, sim.stepcnt, sim.food_in_hub(),
            phenotypes[0]
            )
        simresults.save_to_file()

    # print("Total food in the hub", len(food_objects))
    value = sim.food_in_hub()

    foraging_percent = (
        value * 100.0) / (sim.num_agents * 2.0)

    sucess = False
    print ('Foraging percent', value)

    if foraging_percent >= threshold:
        print('Foraging success')
        sucess = True

    # sim.experiment.update_experiment_simulation(value, sucess)
    sim.experiment.update_experiment_simulation(value, sucess)

    # Plot the fitness in the graph
    graph = GraphACC(sim.pname, 'simulation.csv')
    graph.gen_plot()


def evolve(iteration):
    """Learning Algorithm block."""
    # iteration = 10000

    env = EnvironmentModel(50, 100, 100, 10, iter=iteration)
    env.build_environment_from_json()

    # for all agents store the information about hub
    for agent in env.agents:
        agent.shared_content['Hub'] = {env.hub}

    # Hub and site object
    # print(env.hub, env.site)

    # Iterate and execute each step in the environment
    for i in range(iteration):
        env.step()

    env.experiment.update_experiment()

    best_agent = env.top

    # Find if food has been deposited in the hub
    grid = env.grid
    food_loc = (0, 0)
    neighbours = grid.get_neighborhood(food_loc, 10)
    food_objects = grid.get_objects_from_list_of_grid('Food', neighbours)
    print ('Total food in the hub', len(food_objects))

    for food in food_objects:
        print (food.phenotype)
        if food.agent_name == best_agent:
            print('Foraging success', food.id, food.location)

    # Plot the fitness in the graph
    graph = Graph(env.pname, 'best.csv', ['explore', 'foraging'])
    graph.gen_best_plots()

    # Test the evolved behavior
    return env.agents  # [best_agent]


def main(iter):
    """Block for the main function."""
    # agents = evolve(iter)
    # simulate(agents, iter)
    phenotype = ['<?xml version=\"1.0\" encoding=\"UTF-8\"?><Sequence><Sequence><Sequence><Selector><cond>NeighbourObjects</cond><act>CompositeSingleCarry_Food</act></Selector><Sequence><cond>NeighbourObjects</cond><act>Explore</act></Sequence></Sequence> <Selector><cond>NeighbourObjects</cond><cond>NeighbourObjects</cond><act>MoveTowards_Hub</act></Selector></Sequence><Sequence><Sequence><Sequence><cond>IsDropable_Hub</cond><act>CompositeDrop_Food</act></Sequence> <Sequence><cond>NeighbourObjects</cond><act>CompositeSingleCarry_Food</act></Sequence></Sequence> <Selector><cond>IsDropable_Hub</cond><act>MoveAway_Sites</act></Selector></Sequence></Sequence>']
    simulate(phenotype, iter)


if __name__ == '__main__':
    # Running 50 experiments in parallel

    # Parallel(n_jobs=8)(delayed(main)(i) for i in range(1000, 100000, 2000))
    # Parallel(n_jobs=4)(delayed(main)(i) for i in range(1000, 8000, 2000))
    main(1800)
