from model import EnvironmentModel, RunEnvironmentModel
# from swarms.utils.jsonhandler import JsonData
from swarms.utils.graph import Graph
from joblib import Parallel, delayed

# Global variables for width and height
width = 100
height = 100

UI = False


def simulate(agent, iteration):
    # Testing the performane of evolved behavior
    phenotype = agent.individual[0].phenotype
    iteration = 10000

    sim = RunEnvironmentModel(100, 100, 100, 10, iter=iteration, xmlstring=phenotype)
    sim.build_environment_from_json()

    # for all agents store the information about hub
    for agent in sim.agents:
        agent.shared_content['Hub'] = {sim.hub}

    # Iterate and execute each step in the environment
    for i in range(iteration):
        sim.step()

    sim.experiment.update_experiment()

    # Find if food has been deposited in the hub
    grid = sim.grid
    food_loc = (0, 0)
    neighbours = grid.get_neighborhood(food_loc, 10)
    food_objects = grid.get_objects_from_list_of_grid('Food', neighbours)
    print("Total food in the hub", len(food_objects))
    foraging_percent = (
        len(food_objects) * 100.0) / (sim.num_agents * 2.0)

    if foraging_percent >= 75.0:
        print('Foraging success')


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
    # main()
