"""Experiment script to run GEESE algorithm for foraging."""

from simmodel import SimForgModel, EvolModel, SimModel

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
        N, width, height, 10, iter=iteration, expname='ForgeEvolve',
        agent=agent, parm='res.txt')
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
        env.phenotypes += list(food.phenotype.keys())

    jfilename = env.pname + '/' + env.runid + '.json'

    JsonPhenotypeData.to_json(env.phenotypes, jfilename)

    # Not using this method right now
    # env.phenotypes = extract_phenotype(env.agents, jfilename)

    # Plot the fitness in the graph
    # graph = Graph(env.pname, 'best.csv', ['explore', 'foraging'])
    # graph.gen_best_plots()

    # Test the evolved behavior
    return env


def simulate(env, iteration):
    """Test the performane of evolved behavior."""
    # phenotype = agent.individual[0].phenotype
    # phenotypes = extract_phenotype(agents)
    phenotypes = env[0]
    print('phenotype length',len(phenotypes))
    threshold = 1.0

    sim = SimModel(
        100, width, height, 10, iter=iteration, xmlstrings=phenotypes, pname=env[1],
        expname='MSFCommSimulate', agent='SimAgent')
    sim.build_environment_from_json()

    # for all agents store the information about hub
    for agent in sim.agents:
        agent.shared_content['Hub'] = {sim.hub}
        # print(agent.bt.behaviour_tree.root)
        # agent.shared_content['Sites'] = {sim.site}

    # print(sim.hub, sim.site, sim.foods)
    simresults = SimulationResults(
        sim.pname, sim.connect, sim.sn, sim.stepcnt, sim.food_in_hub(),
        phenotypes[0]
        )

    simresults.save_phenotype()
    simresults.save_to_file()

    # Iterate and execute each step in the environment
    for i in range(iteration):
        # For every iteration we need to store the results
        # Save them into db or a file
        sim.step()
        simresults = SimulationResults(
            sim.pname, sim.connect, sim.sn, sim.stepcnt, sim.food_in_hub(),
            phenotypes[0]
            )
        simresults.save_to_file()

    print ('food at site', len(sim.food_in_loc(sim.site.location)))
    print ('food at hub', len(sim.food_in_loc(sim.hub.location)))
    # print("Total food in the hub", len(food_objects))

    # print([food.location for food in sim.foods])
    food_objects = sim.food_in_loc(sim.hub.location)

    for food in food_objects:
        print('simulate phenotye:', dir(food))
    value = sim.food_in_hub()

    foraging_percent = (
        value * 100.0) / (sim.num_agents * 2.0)

    sucess = False
    print('Foraging percent', value)

    if foraging_percent >= threshold:
        print('Foraging success')
        sucess = True

    # sim.experiment.update_experiment_simulation(value, sucess)

    # Plot the fitness in the graph
    graph = GraphACC(sim.pname, 'simulation.csv')
    graph.gen_plot()


def main(args):
    # env = evolve(args.iteration)
    env = evolve(args)
    print('Evolution Finished')
    if len(env.phenotypes) >= 1:
        steps = [5000 for i in range(50)]
        env = (env.phenotypes, env.pname)
        print(env)
        for step in steps:
            print('Simulation the evolved phenotypes')
            simulate(env, step)
            # simulate_res1(env, step)
            # simulate_res2(env, step)
        # Parallel(n_jobs=4)(delayed(simulate)(env, i) for i in steps)
        # Parallel(n_jobs=4)(delayed(simulate_res1)(env, i) for i in steps)
        # Parallel(n_jobs=4)(delayed(simulate_res2)(env, i) for i in steps)
        # simulate(env, 10000)
    print('=======End=========')


def readtxt():
    # jfilename = '/tmp/behavior.json'
    # jfilename = '/tmp/1617336611651.json'
    # data = JsonPhenotypeData.load_json_file(jfilename)
    # print(data)

    fo = open("/tmp/swarmnewbehaviors1.txt", "r+")
    # print ("Name of the file: ", fo.name)
    phenotypes = {}
    i = 0
    while True:
        line = fo.readline()
        phenotypes.update(eval(line))
        # print(i, len(phenotypes.keys()))
        # print ("Read Line: %s" % (line))
        i += 1
        if i == 16:
            break
    # {k: d[k] for k in sorted(d, key=d.get)}
    phenotypeslist = [k for k in sorted(phenotypes, key=phenotypes.get, reverse=True)]
    print(phenotypeslist)
    # print(phenotypeslist, len(phenotypeslist))

    # line = fo.readline(5)
    # print ("Read Line: %s" % (line))
    return phenotypeslist

def readjson():
    jfilename = '/home/aadeshnpn/Desktop/sforaging/07777SForagingSimulation/1538473090382007.json'
    data = JsonPhenotypeData.load_json_file(jfilename)    
    return data['phenotypes']

def run_phenotype_exp():
    # phenotypes = readtxt()
    phenotypes = readjson()
    # steps = [5000 for i in range(50)]
    env = (phenotypes, '/tmp/swarm/data/experiment/')
    # for step in steps:
    print('Simulation the evolved phenotypes')
    simulate(env, 5000)



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
    # main(args)
    iterations = list(range(1000, 50000, 2000))
    itlist = []
    for i in iterations:
        itlist += [i] * 16
    print(itlist)
    # Parallel(n_jobs=8)(delayed(main)(i) for i in itlist)
    # readjson()
    run_phenotype_exp()