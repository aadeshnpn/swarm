from swarms.model import EnvironmentModel
from swarms.agent import SwarmAgent

import argparse

import os, sys

## Global variables for width and height
width = 50
height = 50

#sys.path += [os.path.join('/home/aadeshnpn/Documents/BYU/hcmi/swarm/lib')]

print (sys.path)
#exit(1)

def main():
    
    env = EnvironmentModel(10, width, height)

    for i in range(1000):
        env.step()

    for agent in env.schedule.agents:
        print (agent.unique_id, agent.wealth)

if __name__ == '__main__':
    main()