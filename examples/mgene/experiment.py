"""Experiment script to run coevolution simulation."""

import numpy as np
# import pdb
# import hashlib
# import sys
import argparse
import pickle
from model import CombineModel, EvolveModel, SimCoevoModel
from swarms.utils.jsonhandler import JsonPhenotypeData
from swarms.utils.graph import Graph, GraphACC  # noqa : F401
from joblib import Parallel, delayed    # noqa : F401
from swarms.utils.results import SimulationResultsLt, SimulationResults, SimulationResultsWTime
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
    if foraging_percent > 65:
        success = True
        # Save the behavior tree repotires.
        agents = env.agents
        # brepotires = filter_brepotires(agents)
        combine_controllers(args, agents)
    else:
        success = False

    # env.experiment.update_experiment_simulation(foraging_percent, success)
    # # phenotypes = env.behavior_sampling(ratio_value=0.99)
    # # print(phenotypes)
    # phenotypes = filter_agents(env.agents)
    # # phenotypes = [[gene.phenotype for gene in agent.brepotire.values()] for agent in env.agents]
    # JsonPhenotypeData.to_json(phenotypes, env.pname + '/' + env.runid + '_all.json')
    # # csize = compute_controller_size(env.agents)
    # # JsonPhenotypeData.to_json(csize, env.pname + '/' + env.runid + 'shape.json')
    # # print(env.agents[0].brepotire)
    # if foraging_percent > 70:
    #     static_bheavior_test(args, env.agents, env.pname)


def filter_agents(agents):
    # xmlstrings = [[gene.phenotype for gene in agent.brepotire.values()] for agent in agents if len(agent.brepotire.values())>3]
    filteredagents = {}
    for agent in agents:
        if len(agent.brepotire.values()) >=4:
            filteredagents[agent] = np.max([gene.fitness for gene in agent.brepotire.values()])
    sortedagents = dict(sorted(filteredagents.items(), key=lambda item: item[1], reverse=True))
    # return sortedagents
    # sortedagents = list(sortedagents.keys())[: int(len(sortedagents)*ratio)]
    return [[gene.phenotype for gene in sortbehavior(agent.brepotire)] for agent in sortedagents]

    # return [[gene.phenotype for gene in agent.brepotire.values()] for agent in agents]


def sortbehavior(brepotire):
    # names = [
    #    CompositeDrop.__name__, MoveAwayNormal.__name__,
    #    CompositeSingleCarry.__name__, MoveTowardsNormal.__name__,
    #    ExploreNormal.__name__]
    # return dict(sorted(brepotire.items(), key=lambda item: item[0]))
    # print(names)
    # return [brepotire[k] for k in names]
    # brepotire = {gene:gene.fitness for gene in brepotire.values()}
    sortedbehavior = dict(sorted(brepotire.items(), key=lambda item: item[1].fitness))
    return list(sortedbehavior.values())
    # return sortedbehavior


def filter_brepotires(agents):
    return {agent.name:agent.brepotire for agent in agents}


def combine_controllers(args, agents=None, pname='/tmp'):
    # brepotires = filter_brepotires(agents)
    pname = '/tmp/swarm/data/experiments/'
    with open('/tmp/behaviors_16642277014973.pickle', 'rb') as handle:
        brepotires = pickle.load(handle)

    env = CombineModel(
        args.n, width, height, 10, iter=args.iter,
        brepotires=brepotires, args=args)
    env.build_environment_from_json()
    env.create_agents()
    results = SimulationResultsWTime(
            env.pname, env.connect, env.sn, env.stepcnt, env.foraging_percent(), None)
    results.save_to_file()

    for i in range(args.iter):
        env.step()
        results = SimulationResultsWTime(
            env.pname, env.connect, env.sn, env.stepcnt, env.foraging_percent(), None)
        results.save_to_file()

    sorted_agents = sorted(
        env.agents, key=lambda x: x.individual[0].fitness, reverse=True)

    phenotypes = [agent.individual[0].phenotype for agent in sorted_agents]
    sorted_brepotires = [brepotires[agent.name] for agent in sorted_agents]
    print('pheonetype:', len(phenotypes), 'brepotirese', len(sorted_brepotires))
    with open(env.pname +'/behaviors_' + env.runid + '.pickle', 'wb') as handle:
        pickle.dump(sorted_brepotires, handle, protocol=pickle.HIGHEST_PROTOCOL)
    JsonPhenotypeData.to_json(phenotypes, env.pname + '/' + env.runid + '_all.json')
    # print([a.individual[0].phenotype for a in env.agents])

    # Run static behavior experiments
    if env.foraging_percent() > 40:
        static_bheavior_test(args, phenotypes, sorted_brepotires, env.pname)


def static_bheavior_test(args, xmlstringsall, brepotires, pname):
    # xmlstrings = [[gene.phenotype for gene in agent.brepotire.values()] for agent in agents if len(agent.brepotire.values())>3]
    # sortedagentsall = filter_agents(agents)
    # print('sorted agents', len(sortedagentsall))
    print(len(brepotires))
    for sample in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99]:
        # pname = '/tmp/swarm/data/experiments/'+ str(sample) + '/'
        # print(xmlstrings)
        pname_static = pname + '/' + str(sample)
        xmlstrings = xmlstringsall[:int(len(xmlstringsall)*sample)]
        env = SimCoevoModel(
            args.n, width, height, 10, iter=args.iter, xmlstrings=xmlstrings,
            expsite=30, pname=pname_static, brepotires=brepotires)
        env.build_environment_from_json()
        # for agent in env.agents:
        #     agent.shared_content['Hub'] = {env.hub}
        # JsonPhenotypeData.to_json(xmlstrings, pname + '/' + env.runid + '_all.json')
        results = SimulationResults(
            env.pname, env.connect, env.sn, env.stepcnt, env.food_in_hub(), None)
        results.save_to_file()

        for i in range(12000):
            env.step()
            results = SimulationResults(
                env.pname, env.connect, env.sn, env.stepcnt, env.food_in_hub(), None)
            results.save_to_file()
        print('Test foraging percent', env.food_in_hub(), ' ,Sampling:', sample)
        # print([food.location for food in env.foods])


def static_bheavior_test_from_json(args, xmlstringsall=None, brepotires=None, pname=None):
    xmlstringsall = JsonPhenotypeData.load_json_file(args.fname)
    xmlstringsall = xmlstringsall['phenotypes']
    with open('/tmp/behaviors_16644389907579.pickle', 'rb') as handle:
        brepotires = pickle.load(handle)
    print(len(brepotires))
    for sample in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99]:
        pname = '/tmp/swarm/data/experiments/'+ str(sample) + '/'
        # print(xmlstrings)
        pname_static = pname
        xmlstrings = xmlstringsall[:int(len(xmlstringsall)*sample)]
        env = SimCoevoModel(
            args.n, width, height, 10, iter=args.iter, xmlstrings=xmlstrings,
            expsite=30, pname=pname_static, brepotires=brepotires)
        env.build_environment_from_json()
        # for agent in env.agents:
        #     agent.shared_content['Hub'] = {env.hub}
        # JsonPhenotypeData.to_json(xmlstrings, pname + '/' + env.runid + '_all.json')
        results = SimulationResults(
            env.pname, env.connect, env.sn, env.stepcnt, env.food_in_hub(), None)
        results.save_to_file()

        for i in range(12000):
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

def combine_controller_exp(args):
    if args.threads <= 1:
        for i in range(args.runs):
            combine_controllers(args)
    else:
        Parallel(
                n_jobs=args.threads)(
                    delayed(combine_controllers)(args) for i in range(
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
        3: combine_controller_exp
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
