"""Experiment script to run coevolution simulation."""

from copyreg import pickle
from unittest import result
import numpy as np
# import pdb
# import hashlib
# import sys
import argparse
from model import EvolveModel, SimCoevoModel
from swarms.utils.jsonhandler import JsonPhenotypeData
from swarms.utils.graph import Graph, GraphACC  # noqa : F401
from joblib import Parallel, delayed    # noqa : F401
from swarms.utils.results import SimulationResultsLt, SimulationResults
from swarms.behaviors.scbehaviors import (
    ExploreNormal, MoveAwayNormal, MoveTowardsNormal,
    CompositeSingleCarry, CompositeDrop)

# import py_trees
# Global variables for width and height
width = 100
height = 100

UI = False


def learning_phase(args):
    """Learning Algorithm block."""
    iteration = args.iter
    no_agents = args.n
    db = args.db
    # early_stop =False
    threshold = args.threshold
    gstep = args.gstep
    expp = args.expp
    # Evolution environment
    env = EvolveModel(
        no_agents, width, height, 10, iter=iteration, db=db,
        threshold=threshold, gstep=gstep, expp=expp, args=args)
    env.build_environment_from_json()
    env.create_agents()
    # Validation Step parameter
    # Run the validation test every these many steps
    # validation_step = 11000

    # Iterate and execute each step in the environment
    # Take a step i number of step in evolution environment
    # Take a 1000 step in validation environment sampling from the evolution
    # Make the validation envronmnet same as the evolution environment
    ltrateavg, ltratestd = env.compute_lt_rate()
    results = SimulationResultsLt(
        env.pname, env.connect, env.sn, env.stepcnt,
        env.foraging_percent(), None, env.no_agent_dead(),
        ltrateavg, ltratestd, env.compute_genetic_rate(), db=False
        )
    # Save the data in a result csv file
    results.save_to_file()
    # import py_trees
    # for agent in env.agents:
    #     print(
    #         agent.name, py_trees.display.ascii_tree(
    #             agent.bt.behaviour_tree.root))

    for i in range(iteration):
        # Take a step in evolution
        env.step()
        ltrateavg, ltratestd = env.compute_lt_rate()
        results = SimulationResultsLt(
            env.pname, env.connect, env.sn, env.stepcnt,
            env.foraging_percent(), None, env.no_agent_dead(),
            ltrateavg, ltratestd, env.compute_genetic_rate(), db=False
            )
        # Save the data in a result csv file
        results.save_to_file()
        # env.gather_info()

    # Update the experiment table
    foraging_percent = env.foraging_percent()
    if foraging_percent > 5:
        success = True
    else:
        success = False
    env.experiment.update_experiment_simulation(foraging_percent, success)
    # phenotypes = env.behavior_sampling(ratio_value=0.99)
    # print(phenotypes)
    phenotypes = filter_agents(env.agents, ratio=0.999)
    JsonPhenotypeData.to_json(phenotypes, env.pname + '/' + env.runid + '_all.json')
    # csize = compute_controller_size(env.agents)
    # JsonPhenotypeData.to_json(csize, env.pname + '/' + env.runid + 'shape.json')
    # print(env.agents[0].brepotire)
    if foraging_percent > 5:
        static_bheavior_test(args, env.agents, env.pname)


def filter_agents(agents, ratio=0.1):
    # xmlstrings = [[gene.phenotype for gene in agent.brepotire.values()] for agent in agents if len(agent.brepotire.values())>3]
    filteredagents = {}
    for agent in agents:
        if len(agent.brepotire.values()) >=4:
            filteredagents[agent] = np.average([gene.fitness for gene in agent.brepotire.values()])
    sortedagents = dict(sorted(filteredagents.items(), key=lambda item: item[1], reverse=True))
    sortedagents = list(sortedagents.keys())[: int(len(sortedagents)*ratio)]
    return [[gene.phenotype for gene in sortbehavior(agent.brepotire)] for agent in sortedagents]


def sortbehavior(brepotire):
    # names = [
    #    CompositeDrop.__name__, MoveAwayNormal.__name__,
    #    CompositeSingleCarry.__name__, MoveTowardsNormal.__name__,
    #    ExploreNormal.__name__]
    # return dict(sorted(brepotire.items(), key=lambda item: item[0]))
    # print(names)
    # return [brepotire[k] for k in names]
    brepotire = {gene:gene.fitness for gene in brepotire.values()}
    sortedbehavior = dict(sorted(brepotire.items(), key=lambda item: item[1]))
    return list(sortedbehavior.keys())


def static_bheavior_test(args, agents, pname):
    # xmlstrings = [[gene.phenotype for gene in agent.brepotire.values()] for agent in agents if len(agent.brepotire.values())>3]
    for sample in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99]:
        pname = pname + '/' + str(sample)
        xmlstrings = filter_agents(agents, ratio=sample)
        # print(xmlstrings)
        env = SimCoevoModel(
            args.n, width, height, 10, iter=args.iter, xmlstrings=xmlstrings,
            expsite=30, pname=pname)
        env.build_environment_from_json()
        JsonPhenotypeData.to_json(
            xmlstrings, pname + '/' + env.runid + '_all_' + str(sample) + '_.json')
        results = SimulationResults(
            env.pname, env.connect, env.sn, env.stepcnt, env.food_in_hub(), None)
        results.save_to_file()

        for i in range(3000):
            env.step()
            results = SimulationResults(
                env.pname, env.connect, env.sn, env.stepcnt, env.food_in_hub(), None)
            results.save_to_file()
        print('Test foraging percent:', env.food_in_hub(), ' ,Sampling:', sample)


def static_bheavior_test_from_json(args):
    xmlstrings = JsonPhenotypeData.load_json_file(args.fname)
    xmlstrings = xmlstrings['phenotypes']
    print(len(xmlstrings))
    for sample in [1.0]: #, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99]:
        pname = '/tmp/swarm/data/experiments/'+ str(sample) + '/'
        # print(xmlstrings)
        xmlstrings = xmlstrings[:int(len(xmlstrings)*sample)]
        env = SimCoevoModel(
            args.n, width, height, 10, iter=args.iter, xmlstrings=xmlstrings,
            expsite=30, pname=pname)
        env.build_environment_from_json()
        for agent in env.agents:
            agent.shared_content['Hub'] = {env.hub}
        # JsonPhenotypeData.to_json(xmlstrings, pname + '/' + env.runid + '_all.json')
        results = SimulationResults(
            env.pname, env.connect, env.sn, env.stepcnt, env.food_in_hub(), None)
        results.save_to_file()

        for i in range(5000):
            env.step()
            results = SimulationResults(
                env.pname, env.connect, env.sn, env.stepcnt, env.food_in_hub(), None)
            results.save_to_file()
        print('Test foraging percent', env.food_in_hub(), ' ,Sampling:', sample)
        # print([food.location for food in env.foods])


def compute_controller_size(agents):
    sizedict = dict()
    for agent in agents:
        allnodes = list(agent.bt.behaviour_tree.root.iterate())
        # print(agent.name, [node.name for node in allnodes ])
        actions = list(filter(
            lambda x: x.name.split('_')[-1] == 'Act', allnodes)
            )
        # print(actions[0].status, type(actions[0]).__name__)
        sizedict[len(actions)] = sizedict.get(len(actions), 0) + 1
    return sizedict


def phenotype_to_json(pname, runid, phenotypes):
    """Store the phenotype to a json file."""
    jfilename = pname + '/' + runid + '.json'
    JsonPhenotypeData.to_json(phenotypes, jfilename)


def exp_evol(iter, n, db):
    """Block for the main function."""
    # Run the evolutionary learning algorithm
    phenotypes = learning_phase(iter, n, db)    # noqa : F841
    # learning_phase(iter)
    # Run the evolved behaviors on a test environment
    # if phenotypes is not None:
    #     test_loop(phenotypes, 5000)


def standard_evolution(args):
    # phenotypes = learning_phase(iter, n, db)
    if args.threads <= 1:
        for i in range(args.runs):
            learning_phase(args)
    else:
        Parallel(
                n_jobs=args.threads)(
                    delayed(learning_phase)(args) for i in range(
                        args.runs))


def static_behavior_test(args):
    # static_bheavior_test_from_json,
    if args.threads <= 1:
        for i in range(args.runs):
            static_bheavior_test_from_json(args)
    else:
        Parallel(
                n_jobs=args.threads)(
                    delayed(static_bheavior_test_from_json)(args) for i in range(
                        args.runs))


def experiments(args):
    ## New experiments      # noqa : E266
    ## Remove the communication behavior nodes  # noqa : E266
    ## Increase the size of the traps and obstacles     # noqa : E266
    ## Increase the size of the world   # noqa : E266
    exp_no = {
        0: standard_evolution,
        1: exp_varying_n_evolution,
        2: static_behavior_test,
        # 3: single_evo,
        # 4: behavior_sampling_after,
        # 5: exp_with_size_trap
    }
    exp_no[args.exp_no](args)


def exp_varying_n_evolution(args):
    for n in [100, 150, 200]:
        Parallel(
            n_jobs=args.threads)(
                delayed(
                    exp_evol)(args.iter, n, args.db) for i in range(args.runs))
    # for i in range(args.runs):
    #     exp_evol_sample(args.iter, 100, args.db)


# def behavior_sampling(args):
#     exp_evol_sample(args.iter, 50, args.db, args.runs, args.threads)


def single_evo(args):
    exp_evol(args.iter, 50, False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--exp_no', default=0, type=int)
    parser.add_argument('--runs', default=1, type=int)
    parser.add_argument('--threads', default=1, type=int)
    parser.add_argument('--iter', default=12000, type=int)
    parser.add_argument('--db', default=False, type=bool)
    parser.add_argument('--threshold', default=10, type=int)
    parser.add_argument('--gstep', default=200, type=int)
    parser.add_argument('--expp', default=2, type=int)
    parser.add_argument('--n', default=50, type=int)
    parser.add_argument(
        '--addobject', default=None, choices=[
            None, 'Obstacles', 'Traps', 'Hub', 'Sites'], type=str)
    parser.add_argument(
        '--removeobject', default=None, choices=[
            None, 'Obstacles', 'Traps', 'Hub', 'Sites'], type=str)
    parser.add_argument(
        '--moveobject', default=None, choices=[
            None, 'Obstacles', 'Traps', 'Hub', 'Sites'], type=str)
    parser.add_argument(
        '--jamcommun', default=None, choices=[None, 'Cue', 'Signal'], type=str)
    parser.add_argument('--probability', default=0.5, type=float)
    parser.add_argument('--no_objects', default=1, type=int)
    parser.add_argument('--location', default=(-np.inf, -np.inf), type=str)
    parser.add_argument(
        '--fname',
        default='/home/aadeshnpn/Desktop/mgene/16512150699454EvoCoevolutionPPA/16512205317773_all.json',
        type=str)
    parser.add_argument('--radius', default=5, type=int)
    parser.add_argument('--time', default=10000, type=int)
    parser.add_argument('--stoplen', default=0, type=int)
    parser.add_argument('--iprob', default=0.85, type=float)
    parser.add_argument('--stop_interval', default=1000, type=int)
    parser.add_argument('--no_debris', default=10, type=int)
    args = parser.parse_args()
    args.stoplen = args.time + args.stop_interval
    print(args)
    experiments(args)
