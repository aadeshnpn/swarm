"""Experiment script to run GEESE algorithm for foraging."""

from simmodel import SimForgModel, EvolModel

# from swarms.utils.jsonhandler import JsonData
from swarms.utils.graph import GraphACC
from joblib import Parallel, delayed    # noqa : F401
from swarms.utils.results import SimulationResults
from swarms.utils.jsonhandler import JsonPhenotypeData
from simagent import SimForgAgentWith, SimForgAgentWithout
import argparse
import os
import pathlib
# Global variables for width and height
width = 100
height = 100

UI = False


def evolve(iteration, agent='EvolAgent', N=100):
    """Learning Algorithm block."""
    # iteration = 10000

    env = EvolModel(
        N, 200, 200, 10, iter=iteration, expname='ForgeEvolve',
        agent=agent, parm='swarm_comm.txt')
    env.build_environment_from_json()

    # for all agents store the information about hub
    for agent in env.agents:
        agent.shared_content['Hub'] = {env.hub}

    # Hub and site object
    # print(env.hub, env.site)

    # Iterate and execute each step in the environment
    for i in range(iteration):
        env.step()

    # env.experiment.update_experiment()

    # Collecting phenotypes on the basis of food collected
    # Find if food has been deposited in the hub

    food_objects = env.food_in_loc(env.hub.location)
    # print('Total food in the hub evolution:', len(food_objects))
    env.phenotypes = []
    for food in food_objects:
        print(food.phenotype)
        env.phenotypes += list(food.phenotype.values())

    jfilename = env.pname + '/' + env.runid + '.json'

    JsonPhenotypeData.to_json(env.phenotypes, jfilename)

    # Not using this method right now
    # env.phenotypes = extract_phenotype(env.agents, jfilename)

    # Plot the fitness in the graph
    # graph = Graph(env.pname, 'best.csv', ['explore', 'foraging'])
    # graph.gen_best_plots()

    # Test the evolved behavior
    return env


def main(args):
    evolve(args.iteration)


if __name__ == '__main__':
    # Running 50 experiments in parallel
    # steps = [100000 for i in range(50)]
    # Parallel(n_jobs=8)(delayed(main)(i) for i in steps)
    # Parallel(n_jobs=16)(delayed(main)(i) for i in range(1000, 100000, 2000))
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--n', default=100, type=int)
    # [SimForgAgentWith, SimForgAgentWithout])        
    parser.add_argument('--agent', default=0, choices=[0, 1])
    parser.add_argument('--runs', default=50, type=int)
    parser.add_argument('--iteration', default=6000, type=int)    
    parser.add_argument('--all', default=False)
    args = parser.parse_args()
    print(args)
    main(args)