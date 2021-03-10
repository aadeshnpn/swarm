"""Main script that is called from UI."""

from simmodel import SimForgModel
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
    env = SimForgModel(
        width, height, 100, 10, iter=iteration, xmlstrings=[123], pname='/tmp/', viewer=True)
    env.build_environment_from_json()

    # for all agents store the information about hub
    for agent in env.agents:
        agent.shared_content['Hub'] = {env.hub}

    # Overiding the default viewer properties
    env.ui = UI(
        (width, height), [env.hub], env.agents,
        [env.site], food=env.foods, traps=env.traps)
    
    # Iterate and execute each step in the environment
    for i in range(iteration):
        env.step()

    # print('Execution done')
    # Find if food has been deposited in the hub
    # print('Cleaning Percentage', env.foraging_percent())
    #   print(len(env.debris_cleaned()))

if __name__ == '__main__':
    main()
