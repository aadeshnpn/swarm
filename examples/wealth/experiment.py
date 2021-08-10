from model import WealthEnvironmentModel
from swarms.utils.jsonhandler import JsonPhenotypeData
from swarms.utils.ui import UI

# from joblib import Parallel, delayed
# Global variables for width and height
width = 500
height = 500
viewer = True


def main():
    """Block main."""
    iteration = 2500

    # Create a test environment to visualize
    env = WealthEnvironmentModel(
        1, width, height, 10, seed=None)

    # for all agents store the information about hub
    for agent in env.agents:
        agent.shared_content['Hub'] = {env.hub}

    # Iterate and execute each step in the environment
    for i in range(iteration):
        env.step()


if __name__ == '__main__':
    main()
