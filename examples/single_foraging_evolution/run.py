"""Main script that is called from UI."""

from model import EnvironmentModel
# from swarms.utils.jsonhandler import JsonData
from swarms.utils.graph import Graph

# Global variables for width and height
width = 100
height = 100

UI = False


def main():
    iteration = 10

    env = EnvironmentModel(1, 100, 100, 10, iter=iteration)
    env.build_environment_from_json()

    # for all agents store the information about hub
    for agent in env.agents:
        agent.shared_content['Hub'] = {env.hub}

    # Hub and site object
    print(env.hub, env.site)

    # Iterate and execute each step in the environment
    for i in range(iteration):
        env.step()

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

    """
    for agent in env.agents:
        print(agent.name, agent.food_collected)

        if UI:
            hub = env.render.objects['hub']
            sites = env.render.objects['sites']
            obstacles = env.render.objects['obstacles']
            # traps = env.render.objects['traps']
            derbis = env.render.objects['derbis']
            agents = env.agents
            print(JsonData.to_json(
                width, height, hub, sites, obstacles, None, None, None,
                derbis, agents))
    """


if __name__ == '__main__':
    main()
