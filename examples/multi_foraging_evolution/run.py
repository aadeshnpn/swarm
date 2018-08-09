"""Main script that is called from UI."""

from model import EnvironmentModel
# from swarms.utils.jsonhandler import JsonData
# import time
import py_trees
from swarms.utils.graph import Graph
# Global variables for width and height
width = 100
height = 100

UI = False


def main():
    env = EnvironmentModel(100, 100, 100, 10)
    env.build_environment_from_json()
    for agent in env.agents:
        agent.shared_content['Hub'] = {env.hub}
    print(env.hub, env.site)

    grid = env.grid
    food_loc = (0, 0)
    neighbours = grid.get_neighborhood(food_loc, 60)
    food_objects = grid.get_objects_from_list_of_grid('Food', neighbours)
    # print ('TOtal Food prev', len(food_objects))

    for i in range(10):
        env.step()
        # best = env.find_higest_performer()
        # best = env.find_higest_food_collector()
        # print ('-----------', i)

        # if best.food_collected > 0:
        # print (i, best.name, best.individual[0].fitness, best.food_collected, best.bt.behaviour_tree)
        # output = py_trees.display.ascii_tree(best.bt.behaviour_tree.root)
        # print (output)
    for agent in env.agents:
        if len(agent.attached_objects) > 0:
            output = py_trees.display.ascii_tree(agent.bt.behaviour_tree.root)
            print(agent.name, agent.attached_objects, output)
    # for agent in env.agents:
    #     print (agent.name, agent.attached_objects)
        #if len(env.detect_food_moved()) < 50 and len(env.detect_food_moved()) != 0:
        #    print ('food moved', i, len(env.detect_food_moved()))
        # output = py_trees.display.ascii_tree(best.bt.behaviour_tree.root)
        # print (i, best.name, best.individual[0].fitness, output)

        # time.sleep(0.1)
    # print ('food remaining in site', i, len(env.detect_food_moved()))

    grid = env.grid
    food_loc = (0, 0)
    neighbours = grid.get_neighborhood(food_loc, 5)
    food_objects = grid.get_objects_from_list_of_grid('Food', neighbours)
    for food in food_objects:
        print('food', food.id, food.location)

    graph = Graph(env.pname, 'best.csv', ['diversity', 'explore'])
    graph.gen_best_plots()

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
