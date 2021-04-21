"""Main script that is called from UI."""

from simmodel import SimForgModel
from swarms.utils.jsonhandler import JsonPhenotypeData
from swarms.utils.ui import UI
from simagent import SimForgAgentWithout, SimForgAgentWith
# from joblib import Parallel, delayed
# Global variables for width and height
width = 400
height = 400
viewer = True


def main():
    """Block main."""
    iteration = 2500

    # Create a test environment to visualize
    env = SimForgModel(
        50, width, height, 10, iter=iteration, xmlstrings=[123], 
        pname='/tmp/', viewer=True, agent=SimForgAgentWithout, 
        expsite={"x":50, "y":-50, "radius":10, "q_value":0.9})
    env.build_environment_from_json()

    # for all agents store the information about hub
    for agent in env.agents:
        agent.shared_content['Hub'] = {env.hub}

    # Overiding the default viewer properties
    env.ui = UI(
        (width, height), [env.hub], env.agents,
        [env.site], food=env.foods, traps=[], obstacles=[env.obstacles])
    
    # Iterate and execute each step in the environment
    for i in range(iteration):
        env.step()

    # print('Execution done')
    # Find if food has been deposited in the hub
    # print('Cleaning Percentage', env.foraging_percent())
    #   print(len(env.debris_cleaned()))
    # print ('food at site', len(env.food_in_loc(env.site.location)))
    # print ('food at hub', len(env.food_in_loc(env.hub.location)))


if __name__ == '__main__':
    main()
