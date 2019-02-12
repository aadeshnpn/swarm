"""Main script that is called from UI."""

from model import ViewerModel
from swarms.utils.jsonhandler import JsonPhenotypeData
from swarms.utils.ui import UI
# from joblib import Parallel, delayed
# Global variables for width and height
width = 400
height = 400
viewer = True


def main():
    """Block main."""
    iteration = 5000

    # Create a test environment to visualize
    env = ViewerModel(100, width, height, 10, iter=iteration, viewer=viewer)

    # Build environment from json
    env.build_environment_from_json()

    # Load a json file containing the phenotype
    pfileloc = '/home/aadeshnpn/Documents/BYU/hcmi/hri/thesis/sf/'
    jname = pfileloc + '1538473090382007.json'

    phenotypes = JsonPhenotypeData.load_json_file(jname)['phenotypes']
    # Create the agents in the environment from the sampled behaviors
    # print (len(phenotypes))
    env.create_agents(phenotypes=phenotypes[:50])
    # Overiding the default viewer properties
    env.ui = UI(
        (width, height), [env.hub], env.agents,
        [env.site], food=env.foods)
    # Iterate and execute each step in the environment
    for i in range(iteration):
        env.step()

    # print('Execution done')
    # Find if food has been deposited in the hub
    print('foraging percent', env.foraging_percent())
    grid = env.grid
    neighbours = grid.get_neighborhood(env.hub.location, env.hub.radius)
    # ExecutingAgent
    # food_objects = grid.get_objects_from_list_of_grid('Food', neighbours)
    # agents = grid.get_objects_from_list_of_grid('ExecutingAgent', neighbours)
    # for agent in agents:
    #    print (agent.attached_objects)
    #    print (agent.xmlstring)
    # for food in food_objects:
    #    print('food', food.id, food.location)


if __name__ == '__main__':
    main()
