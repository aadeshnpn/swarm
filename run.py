"""Main script that is called from UI."""

from swarms.model import EnvironmentModel
from swarms.utils.jsonhandler import JsonData
import time

# Global variables for width and height
width = 1600
height = 800

UI = True


def main():
    env = EnvironmentModel(2, 1600, 800, 10, 123)
    for i in range(290002):
        env.step()
        time.sleep(0.1)
        if UI:
            hub = env.render_jsondata.objects['hub']
            sites = env.render_jsondata.objects['sites']
            obstacles = env.render_jsondata.objects['obstacles']
            traps = env.render_jsondata.objects['traps']
            derbis = env.render_jsondata.objects['derbis']
            agents = env.agents
            print (JsonData.to_json(width, height, hub, sites, obstacles,
                traps, None, None, derbis, agents))

            # print (JsonData.to_json(width, height, hub, [], [],
            #    [], None, None, derbis, agents))


if __name__ == '__main__':
    main()
