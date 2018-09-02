"""Main script that is called from UI."""

from model import EnvironmentModel


# Global variables for width and height
width = 100
height = 100

UI = False


def main():
    iteration = 100000

    env = EnvironmentModel(100, 100, 100, 10, iter=iteration)
    env.build_environment_from_json()

    # for all agents store the information about hub
    for agent in env.agents:
        agent.shared_content['Hub'] = {env.hub}

    # Iterate and execute each step in the environment
    # print ('Step', 'Name', 'TS', 'DEL', 'OVF', 'EXP', 'CAR', 'FOR', 'GNM')
    for i in range(iteration):
        env.step()

    print (len(env.food_in_loc(env.hub.location)))


if __name__ == '__main__':
    main()
