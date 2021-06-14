"""Main script that is called from UI."""

from model import ViewerModel
from swarms.utils.jsonhandler import JsonPhenotypeData

# from joblib import Parallel, delayed
# Global variables for width and height
width = 800
height = 800
viewer = True


def main():
    """Block main."""
    iteration = 2500
    # jname = '/tmp/16235340355923-10999.json'
    # jname = '/tmp/16235342688381-4999.json'
    jname = '/tmp/16235346558663.json'
    phenotype = JsonPhenotypeData.load_json_file(jname)['phenotypes']
    # Create a test environment to visualize
    viewer = ViewerModel(
        100, width, height, 10, iter=iteration, viewer=True)
    # Build the environment
    viewer.build_environment_from_json()
    # Create the agents in the environment from the sampled behaviors
    viewer.create_agents(phenotypes=[phenotype[0]], random_init=True)

    # for all agents store the information about hub
    for agent in viewer.agents:
        agent.shared_content['Hub'] = {viewer.hub}

    # Iterate and execute each step in the environment
    for i in range(iteration):
        viewer.step()

    # print('Execution done')
    # Find if food has been deposited in the hub
    # print('Cleaning Percentage', env.foraging_percent())
    #   print(len(env.debris_cleaned()))
    # print ('food at site', len(env.food_in_loc(env.site.location)))
    # print ('food at hub', len(env.food_in_loc(env.hub.location)))


if __name__ == '__main__':
    main()
