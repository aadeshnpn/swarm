"""Main script that is called from UI."""

from model import EnvironmentModel
# from swarms.utils.jsonhandler import JsonData
from swarms.utils.graph import Graph
from joblib import Parallel, delayed

# Global variables for width and height
width = 100
height = 100

UI = False


def main():
    iteration = 10
    # Build the environment
    env = EnvironmentModel(1, 100, 100, 10, iter=iteration)
    env.build_environment_from_json()

    # For all agents register hub
    for agent in env.agents:
        agent.shared_content['Hub'] = {env.hub}

    # Hub and site object
    print(env.hub, env.site)

    # Move the simulation forward
    for i in range(iteration):
        env.step()

    # Update the end time of the experiment
    env.experiment.update_experiment()

    # Find if food has been deposited in the hub
    grid = env.grid
    food_loc = (0, 0)
    neighbours = grid.get_neighborhood(food_loc, 5)
    food_objects = grid.get_objects_from_list_of_grid('Food', neighbours)
    for food in food_objects:
        print('food', food.id, food.location)

    # Plot the fitness in the graph
    graph = Graph(env.pname, 'best.csv', ['diversity', 'explore'])
    graph.gen_best_plots()


if __name__ == '__main__':
    Parallel(n_jobs=4)(delayed(main)() for i in range(1, 50))
    # main()
