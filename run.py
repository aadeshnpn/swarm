"""Main script that is called from UI."""

from swarms.model import EnvironmentModel
from swarms.utils.jsonhandler import JsonData
import time

# Global variables for width and height
width = 1600
height = 800

UI = False


def main():
    env = EnvironmentModel(100, 1600, 800, 10, 123)
    env.build_environment_from_json()
    print(env.hub, env.site)

    for i in range(10000):
        env.step()
        # time.sleep(0.1)

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

            # print (JsonData.to_json(width, height, hub, [], [],
            #    [], None, None, derbis, agents))


if __name__ == '__main__':
    main()
