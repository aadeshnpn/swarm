"""Main script to run."""

from model import RandomWalkSwarmEnvironmentModel
from swarms.utils.jsonhandler import JsonData
import time

width = 1000
height = 800

UI = True


def main():
    """Create the environment.

    Custom environment is created and experiment is ran.
    """
    env = RandomWalkSwarmEnvironmentModel(100, width, height, 10, 123)
    for i in range(100000):
        time.sleep(0.1)
        env.step()
        ui_step(env)


def ui_step(env):
    """UI function.

    Sending information to UI in a JSON format.
    """
    for agent in env.agents:
        if UI:
            # hub = env.render.objects['hub']
            # sites = env.render.objects['sites']
            hub = [env.hub]
            sites = [env.site]
            # obstacles = env.render.objects['obstacles']
            # traps = env.render.objects['traps']
            # derbis = env.render.objects['derbis']
            agents = env.agents
            print(JsonData.to_json(
                width, height, hub, sites, None, None, None, None,
                None, agents))


if __name__ == '__main__':
    main()
