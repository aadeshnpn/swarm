"""Main script that is called from UI."""

from model import ViewerModel
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
    env = ViewerModel(100, width, height, 10, iter=iteration, viewer=viewer)

    # Build environment from json
    env.build_environment_from_json()

    # Load a json file containing the phenotype
    pfileloc = '/home/aadeshnpn/Documents/BYU/hcmi/hri/nest_maint/1539014820252NestM/'
    jname = pfileloc + '1539014820252.json'

    phenotypes = JsonPhenotypeData.load_json_file(jname)['phenotypes']
    # Create the agents in the environment from the sampled behaviors
    print (len(phenotypes))

    env.create_agents(phenotypes=phenotypes)
    # Overiding the default viewer properties
    env.ui = UI(
        (width, height), [env.hub], env.agents,
        [], food=[], debris=env.debris)
    # Iterate and execute each step in the environment
    for i in range(iteration):
        env.step()

    # print('Execution done')
    # Find if food has been deposited in the hub
    print('Cleaning Percentage', env.foraging_percent())
    print(len(env.debris_cleaned()))

if __name__ == '__main__':
    main()
