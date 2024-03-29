"""Experiment script to run coevolution simulation."""

import numpy as np
# import pdb
# import hashlib
# import sys
import argparse
from model import EvolveModel
from swarms.utils.jsonhandler import JsonPhenotypeData
from swarms.utils.graph import Graph, GraphACC  # noqa : F401
from joblib import Parallel, delayed    # noqa : F401
from swarms.utils.results import SimulationResultsLt
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
    # threshold = args.threshold
    # Evolution environment
    env = EvolveModel(
        no_agents, width, height, 10, iter=iteration, db=db,
        threshold=10, gstep=200, expp=0, args=args)
    env.build_environment_from_json()
    env.create_agents(random_init=False)
    env.update_static_objects(i=0)
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
    # env.plot_locality(i=env.stepcnt-1)
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
        # env.plot_locality(i=env.stepcnt-1)
        # env.gather_info()
    # env.plot_locality_all(env.reversemapping)
    np.save(env.pname +
        '/' + str(env.runid) + '_locality.npy',
        np.array([env.locality, env.reversemapping], dtype=object))
    np.save(env.pname +
        '/' + str(env.runid) + '_visual.npy',
        np.array([env.visual, env.reversemapping], dtype=object))
    np.save(env.pname +
        '/' + str(env.runid) + '_staticobjs.npy',
        np.array(env.static_objects, dtype=object))

    # Update the experiment table
    foraging_percent = env.foraging_percent()
    if foraging_percent > 5:
        success = True
    else:
        success = False
    env.experiment.update_experiment_simulation(foraging_percent, success)

    # print('max, min generations', np.max(generations), np.min(generations))
    # pdb.set_trace()
    # allphenotypes = env.behavior_sampling_objects(ratio_value=1.0)
    # # save the phenotype to json file
    # phenotype_to_json(
    #     env.pname, env.runid + '-' + 'all', allphenotypes)

    # print('total generation', [agent.generation for agent in env.agents])
    # try:
    #     # return list(phenotypes.keys())
    #     return allphenotypes, foraging_percent, env.pname
    # except UnboundLocalError:
    #     return None, None, None


def experiments(args):
    ## New experiments      # noqa : E266
    if args.threads == 1:
        for i in range(args.runs):
            learning_phase(args)
    else:
        Parallel(
            n_jobs=args.threads)(
                delayed(
                    learning_phase)(args) for i in range(args.runs))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--runs', default=1, type=int)
    parser.add_argument('--threads', default=1, type=int)
    parser.add_argument('--iter', default=12000, type=int)
    parser.add_argument('--db', default=False, type=bool)
    parser.add_argument('--n', default=50, type=int)
    args = parser.parse_args()
    print(args)
    experiments(args)