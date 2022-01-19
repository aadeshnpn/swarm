"""Experiment script to run coevolution simulation."""

import numpy as np
# import pdb
# import hashlib
import sys
import argparse
from model import EvolveModel
from swarms.utils.jsonhandler import JsonPhenotypeData
from swarms.utils.graph import Graph, GraphACC  # noqa : F401
from joblib import Parallel, delayed    # noqa : F401
from swarms.utils.results import SimulationResultsLt, Results
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
    early_stop =False
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
    validation_step = 11000

    # Iterate and execute each step in the environment
    # Take a step i number of step in evolution environment
    # Take a 1000 step in validation environment sampling from the evolution
    # Make the validation envronmnet same as the evolution environment
    results = SimulationResultsLt(
        env.pname, env.connect, env.sn, env.stepcnt,
        env.foraging_percent(), None, env.maintenance_percent(),
        env.compute_lt_rate(), db=False
        )
    # Save the data in a result csv file
    results.save_to_file()

    for i in range(iteration):
        # Take a step in evolution
        env.step()
        results = SimulationResultsLt(
            env.pname, env.connect, env.sn, env.stepcnt,
            env.foraging_percent(), None, env.maintenance_percent(),
            env.compute_lt_rate(), db=False
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


def phenotype_to_json(pname, runid, phenotypes):
    """Store the phenotype to a json file."""
    jfilename = pname + '/' + runid + '.json'
    JsonPhenotypeData.to_json(phenotypes, jfilename)


def exp_evol(iter, n, db):
    """Block for the main function."""
    # Run the evolutionary learning algorithm
    phenotypes = learning_phase(iter, n, db)
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
                n_jobs=args.threads)(delayed(learning_phase)(args
                    ) for i in range(args.runs))


def experiments(args):
    ## New experiments
    ## Remove the communication behavior nodes
    ## Increase the size of the traps and obstacles
    ## Increase the size of the world
    exp_no = {
        0: standard_evolution,
        1: exp_varying_n_evolution,
        # 2: behavior_sampling,
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
        '--exp_no', default=1, type=int)
    parser.add_argument('--runs', default=36, type=int)
    parser.add_argument('--threads', default=18, type=int)
    parser.add_argument('--iter', default=12000, type=int)
    parser.add_argument('--db', default=False, type=bool)
    parser.add_argument('--threshold', default=10, type=int)
    parser.add_argument('--gstep', default=200, type=int)
    parser.add_argument('--expp', default=2, type=int)
    parser.add_argument('--n', default=50, type=int)
    parser.add_argument('--addobject', default=None, choices= [None, 'Obstacles', 'Traps', 'Hub', 'Sites'], type=str)
    parser.add_argument('--removeobject', default=None, choices= [None, 'Obstacles', 'Traps', 'Hub', 'Sites'], type=str)
    parser.add_argument('--moveobject', default=None, choices= [None, 'Obstacles', 'Traps', 'Hub', 'Sites'], type=str)
    parser.add_argument('--jamcommun', default=None, choices=[None, 'Cue', 'Signal'], type=str)
    parser.add_argument('--probability', default=0.5, type=float)
    parser.add_argument('--no_objects', default=1, type=int)
    parser.add_argument('--location', default=(-np.inf, -np.inf), type=str)
    parser.add_argument('--radius', default=5, type=int)
    parser.add_argument('--time', default=10000, type=int)
    args = parser.parse_args()
    print(args)
    experiments(args)