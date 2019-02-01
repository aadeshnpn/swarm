"""Main script that is called from UI."""

from model import ViewerModel
from swarms.utils.jsonhandler import JsonPhenotypeData
from swarms.utils.ui import UI
# from joblib import Parallel, delayed
# Global variables for width and height
width = 800
height = 800
viewer = True


def main():
    """Block main."""
    iteration = 9000

    # Create a test environment to visualize
    env = ViewerModel(50, width, height, 10, iter=iteration, viewer=viewer)

    # Build environment from json
    env.build_environment_from_json()

    # Load a json file containing the phenotype
    pfileloc = '/home/aadeshnpn/Documents/BYU/hcmi/hri/cooperative_transport/1538447335350COT/'
    jname = pfileloc + '1538447335350.json'

    phenotypes = JsonPhenotypeData.load_json_file(jname)['phenotypes']
    # Create the agents in the environment from the sampled behaviors
    # print (len(phenotypes))
    env.create_agents(phenotypes=phenotypes[30:90])
    # Overiding the default viewer properties
    env.ui = UI(
        (width, height), [env.hub], env.agents,
        [env.site], food=env.foods)
    # Iterate and execute each step in the environment
    for i in range(iteration):
        env.step()

    print ('foraging percent', env.foraging_percent())
    # print('Execution done')
    # Find if food has been deposited in the hub
    grid = env.grid
    neighbours = grid.get_neighborhood(env.hub.location, 10)
    food_objects = grid.get_objects_from_list_of_grid('Food', neighbours)
    # for food in food_objects:
    #    print('food', food.id, food.location)


if __name__ == '__main__':
    main()
