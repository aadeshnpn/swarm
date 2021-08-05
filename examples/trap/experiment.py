
from model import SimTrapModel
from swarms.utils.ui import UI

# from joblib import Parallel, delayed
# Global variables for width and height
width = 400
height = 400
viewer = True


def main():

    # Create a test environment to visualize
    env = SimTrapModel(
        1, width, height, 10)

    # for all agents store the information about hub
    for agent in env.agents:
        agent.shared_content['Hub'] = {env.hub}

    # Overiding the default viewer properties
    # env.ui = UI(
    #     (width, height), [env.hub], env.agents,
    #     [env.target], food=[], traps=[], obstacles=[env.obstacles])

    # Iterate and execute each step in the environment
    for i in range(30):
        env.step()
        print(i, env.agent.location, env.agent.direction)


if __name__  == '__main__':
    main()
