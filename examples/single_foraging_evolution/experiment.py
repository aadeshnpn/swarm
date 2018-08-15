from model import EnvironmentModel, RunEnvironmentModel
# from swarms.utils.jsonhandler import JsonData
from swarms.utils.graph import Graph, GraphACC
from joblib import Parallel, delayed
from swarms.utils.results import SimulationResults

# Global variables for width and height
width = 100
height = 100

UI = False


def simulate(agent, iteration):
    # Testing the performane of evolved behavior
    phenotype = agent.individual[0].phenotype
    # iteration = 10000
    threshold = 75.0
    sim = RunEnvironmentModel(
        100, 100, 100, 10, iter=iteration, xmlstring=phenotype)
    sim.build_environment_from_json()

    # for all agents store the information about hub
    for agent in sim.agents:
        agent.shared_content['Hub'] = {sim.hub}

    simresults = SimulationResults(
        sim.pname, sim.connect, sim.sn, sim.stepcnt, sim.food_in_hub(),
        phenotype
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
            phenotype
            )
        simresults.save_to_file()

    # print("Total food in the hub", len(food_objects))
    value = sim.food_in_hub()

    foraging_percent = (
        value * 100.0) / (sim.num_agents * 2.0)

    sucess = False
    if foraging_percent >= threshold:
        print('Foraging success')
        sucess = True

    # sim.experiment.update_experiment_simulation(value, sucess)
    sim.experiment.update_experiment_simulation(value, sucess)

    # Plot the fitness in the graph
    graph = GraphACC(sim.pname, 'simulation.csv')
    graph.gen_plot()


def evolve(iteration):
    # iteration = 10000

    env = EnvironmentModel(100, 100, 100, 10, iter=iteration)
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
    neighbours = grid.get_neighborhood(food_loc, 5)
    food_objects = grid.get_objects_from_list_of_grid('Food', neighbours)
    for food in food_objects:
        if food.agent_name == best_agent:
            print('Foraging success', food.id, food.location)

    # Plot the fitness in the graph
    graph = Graph(env.pname, 'best.csv', ['explore', 'foraging'])
    graph.gen_best_plots()

    # Test the evolved behavior
    return env.agents[best_agent]


def main(iter):
    agent = evolve(iter)
    simulate(agent, iter)


if __name__ == '__main__':
    # Running 50 experiments in parallel

    # Parallel(n_jobs=4)(delayed(main)(i) for i in range(1000, 900000, 2000))
    Parallel(n_jobs=4)(delayed(main)(i) for i in range(1000, 8000, 2000))
    # main(100)
