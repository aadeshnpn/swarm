
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
        2, width, height, 10, seed=123)

    # for all agents store the information about hub
    for agent in env.agents:
        agent.shared_content['Hub'] = {env.hub}

    # Overiding the default viewer properties
    # env.ui = UI(
    #     (width, height), [env.hub], env.agents,
    #     [env.target], food=[], traps=[], obstacles=[env.obstacles])

    # Iterate and execute each step in the environment
    _, site_grid = env.grid.find_grid(env.target.location)
    for i in range(1000):
        env.step()
        print(i, env.agent.location, env.agent.direction, env.agent.dead, end=' ')
        total_dead = 0
        total_reached_site = 0
        for j in range(len(env.agents)):
            total_dead = total_dead + int(env.agents[j].dead ==True)
            _,agent_grid = env.grid.find_grid(env.agents[j].location)
            total_reached_site = total_reached_site + int(site_grid==agent_grid)
        print(total_dead, total_reached_site)

if __name__  == '__main__':
    main()
