"""Main script that is called from UI."""

from swarms.model import EnvironmentModel
# from swarms.utils.jsonhandler import JsonData
# import time
import py_trees
# Global variables for width and height
width = 100
height = 100

UI = False


def main():
    env = EnvironmentModel(100, 100, 100, 10)
    env.build_environment_from_json()
    print(env.hub, env.site.location)

    for i in range(10):
        env.step()
        best = env.find_higest_performer()
        print (i, best.name, best.individual[0].fitness, best.food_collected, best.bt.behaviour_tree)
        output = py_trees.display.ascii_tree(best.bt.behaviour_tree.root)
        print (output)
        # time.sleep(0.1)
    """
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
    """
            # print (JsonData.to_json(width, height, hub, [], [],
            #    [], None, None, derbis, agents))


if __name__ == '__main__':
    main()
