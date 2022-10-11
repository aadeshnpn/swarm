"""Experiment script to run Nest maintenance simulation."""

import numpy as np
# import pdb
# import hashlib
import sys
import pickle
import argparse
from model import (
    EvolveModel, ValidationModel, TestModel, ViewerModel,
    CombineModel, SimNestMModel)
from swarms.utils.jsonhandler import JsonPhenotypeData
from swarms.utils.graph import Graph, GraphACC  # noqa : F401
from joblib import Parallel, delayed    # noqa : F401
from swarms.utils.results import (
    SimulationResults, SimulationResultsTraps, SimulationResultsLt,
    SimulationResultsWTime)
# import py_trees
# Global variables for width and height
width = 100
height = 100

UI = False


def validation_loop(
        phenotypes, iteration, parentname=None, ratio=1, threshold=10.0, db=False):
    """Validate the evolved behaviors."""
    # Create a validation environment instance
    # print('len of phenotype', len(set(phenotypes)))
    valid = ValidationModel(
        100, width, height, 10, iter=iteration, parent=parentname, ratio=ratio, db=db)
    # print('parent:', parentname, ' children:', valid.runid)
    # Build the environment
    valid.build_environment_from_json()
    # Create the agents in the environment from the sampled behaviors
    phenotypes = valid.behavior_sampling(method='ratio', ratio_value=ratio, phenotype=phenotypes)
    valid.create_agents(phenotypes=phenotypes)
    # print('total food units', valid.total_food_units)
    # Print the BT
    # py_trees.display.print_ascii_tree(valid.agents[1].bt.behaviour_tree.root)
    # py_trees.logging.level = py_trees.logging.Level.DEBUG

    for i in range(iteration):
        valid.step()
        # print ([agent.location for agent in valid.agents])
        validresults = SimulationResults(
            valid.pname, valid.connect, valid.sn, valid.stepcnt,
            valid.foraging_percent(), len(phenotypes), db=db
        )
        validresults.save_to_file()

    # print('food in the hub', valid.agents[0].get_food_in_hub(False))
    # Save the phenotype to json file
    phenotype_to_json(valid.pname, valid.runid + '-' + str(i), phenotypes)

    # Plot the result in the graph
    graph = GraphACC(valid.pname, 'simulation.csv')
    graph.gen_plot()

    # Return true if the sample behavior achieves a threshold
    if valid.foraging_percent() > 5:
        success = True
    else:
        success = False
    valid.experiment.update_experiment_simulation(valid.foraging_percent(), success)

    return success


def test_loop(phenotypes, iteration, parentname=None, ratio=1):
    """Test the phenotypes in a completely different environment."""
    # Create a validation environment instance
    test = TestModel(
        100, width, height, 10, iter=iteration, parent=parentname, ratio=ratio)
    # Build the environment
    test.build_environment_from_json()
    # Create the agents in the environment from the sampled behaviors
    phenotypes = test.behavior_sampling(ratio_value=ratio, phenotype=phenotypes)
    test.create_agents(phenotypes=phenotypes)
    # print(phenotypes)
    # print(test.agents[0].bt)
    # Store the initial result
    testresults = SimulationResultsTraps(
        test.pname, test.connect, test.sn, test.stepcnt,
        test.foraging_percent(), len(phenotypes), test.no_agent_dead()
        )
    # Save the phenotype to a json file
    testresults.save_phenotype()
    # Save the data in a result csv file
    testresults.save_to_file()
    # Save the phenotype of json file
    phenotype_to_json(test.pname, test.runid, phenotypes)
    # Execute the BT in the environment
    for i in range(iteration):
        test.step()

        testresults = SimulationResultsTraps(
            test.pname, test.connect, test.sn, test.stepcnt,
            test.foraging_percent(), len(phenotypes), test.no_agent_dead()
        )
        testresults.save_to_file()

    # Plot the result in the graph
    graph = GraphACC(test.pname, 'simulation.csv')
    graph.gen_plot()
    if test.foraging_percent() > 5:
        success = True
    else:
        success = False
    test.experiment.update_experiment_simulation(test.foraging_percent(), success)
    # print('FP',test.foraging_percent())


def ui_loop(phenotypes, iteration, parentname=None, ratio=1):
    """Test the phenotypes with UI."""
    # Create a viewer environment instance
    viewer = ViewerModel(
        5, width, height, 10, iter=iteration, viewer=True)
    # Build the environment
    viewer.build_environment_from_json()
    phenotypes = viewer.behavior_sampling(ratio=ratio, phenotype=phenotypes)
    # Create the agents in the environment from the sampled behaviors
    viewer.create_agents(phenotypes=phenotypes)
    # print(phenotypes)

    # print(test.agents[0].bt)
    # Store the initial result
    # testresults = SimulationResultsTraps(
    #     test.pname, test.connect, test.sn, test.stepcnt,
    #     test.foraging_percent(), phenotypes[0], test.no_agent_dead()
    #     )
    # # Save the phenotype to a json file
    # testresults.save_phenotype()
    # # Save the data in a result csv file
    # testresults.save_to_file()
    # # Save the phenotype of json file
    # phenotype_to_json(test.pname, test.runid, phenotypes)
    # Execute the BT in the environment
    for i in range(iteration):
        viewer.step()

        # testresults = SimulationResultsTraps(
        #     test.pname, test.connect, test.sn, test.stepcnt,
        #     test.foraging_percent(), phenotypes[0], test.no_agent_dead()
        # )
        # testresults.save_to_file()

    # Plot the result in the graph


def learning_phase(args):
    """Learning Algorithm block."""
    # Evolution environment
    iteration = args.iter
    no_agents = args.n
    db = args.db
    threshold = args.threshold
    gstep = args.gstep
    expp = args.expp
    fitid = args.fitid
    env = EvolveModel(
        no_agents, width, height, 10, iter=iteration, db=db,
        fitid=fitid, threshold=threshold, gstep=gstep, expp=expp,
        args=args)
    env.build_environment_from_json()
    env.create_agents(random_init=True)
    # print([a.get_capacity() for a in env.agents])
    # Validation Step parameter
    # Run the validation test every these many steps
    validation_step = 11000

    # Iterate and execute each step in the environment
    # Take a step i number of step in evolution environment
    # Take a 1000 step in validation environment sampling from the evolution
    # Make the validation envronmnet same as the evolution environment
    ltrateavg, ltratestd = env.compute_lt_rate()
    results = SimulationResultsLt(
        env.pname, env.connect, env.sn, env.stepcnt,
        env.maintenance_percent(), None, env.no_agent_dead(),
        ltrateavg, ltratestd, env.compute_genetic_rate(), db=False
        )
    # Save the data in a result csv file
    results.save_to_file()

    for i in range(iteration):
        # Take a step in evolution
        env.step()
        ltrateavg, ltratestd = env.compute_lt_rate()
        results = SimulationResultsLt(
            env.pname, env.connect, env.sn, env.stepcnt,
            env.maintenance_percent(), None, env.no_agent_dead(),
            ltrateavg, ltratestd, env.compute_genetic_rate(), db=False
            )
        # Save the data in a result csv file
        results.save_to_file()
        # env.gather_info()

    # Update the experiment table
    mpercent = env.maintenance_percent()
    if mpercent > 65:
        success = True
        agents = env.agents
        combine_controllers(args, agents)
    else:
        success = False
    env.experiment.update_experiment_simulation(mpercent, success)


def filter_brepotires(agents):
    return {agent.name:agent.brepotire for agent in agents}


def combine_controllers(args, agents=None, pname='/tmp'):
    brepotires = filter_brepotires(agents)
    # pname = '/tmp/swarm/data/experiments/'
    # with open('/tmp/behaviors_.pickle', 'rb') as handle:
    #     brepotires = pickle.load(handle)

    env = CombineModel(
        args.n, width, height, 10, iter=args.iter,
        brepotires=brepotires, args=args)
    env.build_environment_from_json()
    env.create_agents()
    results = SimulationResults(
            env.pname, env.connect, env.sn, env.stepcnt,
            env.maintenance_percent(), None)
    results.save_to_file()

    for i in range(args.iter):
        env.step()
        results = SimulationResults(
            env.pname, env.connect, env.sn, env.stepcnt,
            env.maintenance_percent(), None)
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
    if env.maintenance_percent() > 40:
        static_bheavior_test_from_json(args, phenotypes, sorted_brepotires, env.pname)


def static_bheavior_test_from_json(args, xmlstringsall=None, brepotires=None, pname=None):
    # xmlstringsall = JsonPhenotypeData.load_json_file(args.fname)
    # xmlstringsall = xmlstringsall['phenotypes']
    # with open('/tmp/behaviors_16642277014973.pickle', 'rb') as handle:
    #     brepotires = pickle.load(handle)
    print(len(brepotires))
    for sample in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99]:
        # pname = '/tmp/swarm/data/experiments/'+ str(sample) + '/'
        # print(xmlstrings)
        pname_static = pname + '/' + str(sample)
        xmlstrings = xmlstringsall[:int(len(xmlstringsall)*sample)]
        env = SimNestMModel(
            args.n, width, height, 10, iter=args.iter, xmlstrings=xmlstrings,
            expsite=30, pname=pname_static, brepotires=brepotires)
        env.build_environment_from_json()
        # for agent in env.agents:
        #     agent.shared_content['Hub'] = {env.hub}
        # JsonPhenotypeData.to_json(xmlstrings, pname + '/' + env.runid + '_all.json')
        results = SimulationResultsWTime(
            env.pname, env.connect, env.sn, env.stepcnt,
            env.maintenance_percent(), None)
        results.save_to_file()

        for i in range(12000):
            env.step()
            results = SimulationResultsWTime(
                env.pname, env.connect, env.sn, env.stepcnt,
                env.maintenance_percent(), None)
            results.save_to_file()
        print('Test maintenance percent',
            env.maintenance_percent(), ' ,Sampling:', sample)
        # print([food.location for food in env.foods])


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


def exp_evol_sample(iter, n, db):
    """Block for the main function."""
    count_exp = 0
    while count_exp <= 15:
        # Run the evolutionary learning algorithm
        phenotypes, fpercent, pname = learning_phase(iter, n, db)
        # learning_phase(iter)
        # Run the evolved behaviors on a test environment
        print('Behavior Sampling experiments', count_exp, fpercent, len(phenotypes))
        if (phenotypes is not None and fpercent >= 80):
            for r in [0.1, 0.2, 0.3, 0.5, 0.7, 1.0]:
                print(r)
                Parallel(
                    n_jobs=2)(delayed(validation_loop)(
                        phenotypes, 5000, pname, r, db) for i in range(40))
            count_exp += 1


def test_json_phenotype(json):
    # jname = '/home/aadeshnpn/Documents/BYU/hcmi/swarm/results/1550083569946511-12000EvoSForge/' + json  # noqa : E501
    # jname = '/tmp/1543367322976111-8000EvoSForge/' + json
    # jname = '/tmp/16235346558663.json'
    jname = '/tmp/16235340355923-10999.json'
    phenotype = JsonPhenotypeData.load_json_file(jname)['phenotypes']
    print(len(phenotype))
    # phenotype = ' '

    #
    test_loop(phenotype, 5000)
    # ui_loop(phenotype, 500)
    validation_loop(phenotype, 5000, '/tmp/swarm/data/experiments/')
    # if validation_loop(phenotype, 2000, 1):
    #    print('foraging success')


def test_top_phenotype(jsonlist):
    #jsonlist = jsonlist.split(' ')
    # jname = '/tmp/1543367322976111-8000EvoSForge/' + json
    phenotypes = []
    for jname in jsonlist:
        phenotype = JsonPhenotypeData.load_json_file(jname)['phenotypes']
        phenotypes.append([phenotype[0]])
    #print (phenotypes[0])
    # test_loop(phenotypes[0], 5000)
    Parallel(
        n_jobs=16)(delayed(test_loop)(
            phenotypes[i], 5000) for i in range(len(phenotypes)))


def test_all_phenotype(idfile='/tmp/experiments/idvalid.txt'):
    # ids = np.genfromtxt(idfile, autostrip=True, unpack=True, dtype=np.int64)
    dirs, ids = np.genfromtxt(idfile, autostrip=True, unpack=True, dtype=np.str_, delimiter=',')
    # print(len(ids), len(dirs))
    # rootdirs = ['/tmp/experiments/50/12000/' + str(id) +'EvoSForgeNew/'+str(id)+'-10999.json' for id in ids]

    rootdirs = [str(dirs[i])+'/'+str(ids[i])+'-4999.json' for i in range(len(ids))]

    phenotypes = []
    # print(rootdirs)
    for jname in rootdirs:
        try:
            phenotype = JsonPhenotypeData.load_json_file(jname)['phenotypes']
            # print(phenotype)
            phenotypes.append(phenotype[0])
        except FileNotFoundError:
            pass

    print(len(phenotypes))

    # Parallel(
    #     n_jobs=4)(delayed(test_loop)(
    #         phenotypes, 5000) for i in range(4))

    Parallel(
        n_jobs=4)(delayed(validation_loop)(
            phenotypes, 5000, '/tmp/swarm/data/experiments/') for i in range(4))


def standard_evolution(args):
    # phenotypes = learning_phase(iter, n, db)
    if args.threads <= 1:
        for i in range(args.runs):
            learning_phase(args)
    else:
        Parallel(
                n_jobs=args.threads)(delayed(learning_phase)(
                    args) for i in range(args.runs))


def experiments(args):
    ## New experiments
    ## Remove the communication behavior nodes
    ## Increase the size of the traps and obstacles
    ## Increase the size of the world
    exp_no = {
        0: standard_evolution,
        1: exp_varying_n_evolution,
        2: behavior_sampling,
        3: single_evo,
        4: behavior_sampling_after,
        # 5: exp_with_size_trap
    }
    exp_no[args.exp_no](args)


def behavior_sampling_after(args):
    jname = '/tmp/experiments/100/12000/1624352990396EvoSForgeNewPPA1/1624352990396-all.json'
    phenotype = JsonPhenotypeData.load_json_file(jname)['phenotypes']
    print(len(phenotype))
    for r in [0.1, 0.2, 0.3, 0.5, 0.7, 1.0]:
        Parallel(
            n_jobs=18)(delayed(validation_loop)(
                phenotype, 5000, None, r, db=False) for i in range(18))

# def exp_with_size_trap(args):
#     jname = '/tmp/experiments/100/12000/1624352990396EvoSForgeNewPPA1/1624352990396-all.json'
#     phenotype = JsonPhenotypeData.load_json_file(jname)['phenotypes']


def exp_varying_n_evolution(args):
    # for n in range(50, 300, 50):
    # Parallel(n_jobs=args.threads)(delayed(exp_evol)(args.iter, n, args.db) for i in range(args.runs))
    for i in range(args.runs):
        exp_evol_sample(args.iter, 100, args.db)


def behavior_sampling(args):
    exp_evol_sample(args.iter, 100, args.db)


def single_evo(args):
    exp_evol(args.iter, 50, False)


if __name__ == '__main__':
    # Running 50 experiments in parallel
    # Parallel(n_jobs=8)(delayed(main)(i) for i in range(2000, 100000, 2000))
    # Parallel(n_jobs=4)(delayed(main)(i) for i in range(1000, 8000, 2000))
    # main(12000)
    # json = '1550083569946511-all.json'
    # test_json_phenotype(None)

    # Parallel(n_jobs=18)(delayed(main)(12000) for i in range(36))
    # main(12000)
    # test_all_phenotype('/tmp/links.txt')
    # jsonlist = sys.argv
    # print ('jsonlist',len(jsonlist))
    # test_top_phenotype(jsonlist[1:])
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--exp_no', default=0, type=int)
    parser.add_argument('--runs', default=36, type=int)
    parser.add_argument('--threads', default=18, type=int)
    parser.add_argument('--iter', default=12000, type=int)
    parser.add_argument('--db', default=False, type=bool)
    parser.add_argument('--fitid', default=3, type=int)
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
    parser.add_argument('--radius', default=5, type=int)
    parser.add_argument('--time', default=10000, type=int)
    parser.add_argument('--stoplen', default=0, type=int)
    parser.add_argument('--iprob', default=0.85, type=float)
    parser.add_argument('--stop_interval', default=1000, type=int)
    args = parser.parse_args()
    args.stoplen = args.time + args.stop_interval
    print(args)
    experiments(args)
