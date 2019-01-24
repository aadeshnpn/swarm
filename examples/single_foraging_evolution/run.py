"""Main script that is called from UI."""

from model import TestModel
from swarms.utils.jsonhandler import JsonPhenotypeData
from swarms.utils.ui import UI
# from joblib import Parallel, delayed
# Global variables for width and height
width = 400
height = 400
viewer = False


def main():
    """Block main."""
    iteration = 5000

    # Create a test environment to visualize
    env = TestModel(200, width, height, 20, iter=iteration, viewer=viewer)

    # Build environment from json
    env.build_environment_from_json()

    # Load a json file containing the phenotype
    pfileloc = '/home/aadeshnpn/Documents/BYU/hcmi/hri/thesis/sf/'
    jname = pfileloc + '1538473090382007.json'

    phenotypes = JsonPhenotypeData.load_json_file(jname)['phenotypes']
    # Create the agents in the environment from the sampled behaviors
    env.create_agents(phenotypes=phenotypes)
    # Overiding the default viewer properties
    env.ui = UI(
        (width, height), [env.hub], env.agents,
        [env.site], food=env.foods)
    # Iterate and execute each step in the environment
    for i in range(iteration):
        env.step()

    # print('Execution done')
    # Find if food has been deposited in the hub
    grid = env.grid
    neighbours = grid.get_neighborhood(env.hub.location, 10)
    food_objects = grid.get_objects_from_list_of_grid('Food', neighbours)
    for food in food_objects:
        print('food', food.id, food.location)

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
