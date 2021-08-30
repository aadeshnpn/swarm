"""Experiment script to run Single source foraging simulation."""

import numpy as np
# import pdb
# import hashlib
import sys
import argparse
from model import EvolveModel, ValidationModel, TestModel, ViewerModel
from swarms.utils.jsonhandler import JsonPhenotypeData
from swarms.utils.graph import Graph, GraphACC  # noqa : F401
from joblib import Parallel, delayed    # noqa : F401
from swarms.utils.results import SimulationResults, SimulationResultsTraps, Results
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


def learning_phase(iteration, no_agents=50, db=False, early_stop=False):
    """Learning Algorithm block."""
    # Evolution environment
    env = EvolveModel(no_agents, width, height, 10, iter=iteration, db=db)
    env.build_environment_from_json()
    env.create_agents()
    # Validation Step parameter
    # Run the validation test every these many steps
    validation_step = 11000

    # Iterate and execute each step in the environment
    # Take a step i number of step in evolution environment
    # Take a 1000 step in validation environment sampling from the evolution
    # Make the validation envronmnet same as the evolution environment
    results = SimulationResultsTraps(
        env.pname, env.connect, env.sn, env.stepcnt,
        env.foraging_percent(), None, env.no_agent_dead(), db=False
        )
    # Save the data in a result csv file
    results.save_to_file()

    for i in range(iteration):
        # Take a step in evolution
        env.step()
        results = SimulationResultsTraps(
            env.pname, env.connect, env.sn, env.stepcnt,
            env.foraging_percent(), None, env.no_agent_dead(), db=False
            )
        # Save the data in a result csv file
        results.save_to_file()
        # env.gather_info()
    # for i in range(iteration):
    #     # Take a step in evolution
    #     env.step()
    #     if (i + 1) % validation_step == 0:
    #         try:
    #             # print([agent.individual[0].fitness for agent in env.agents])
    #             # msg = []
    #             # for agent in env.agents:
    #             #    encode = agent.individual[0].phenotype.encode('utf-8')
    #             #    msg += [(
    #             #        agent.name, hashlib.sha224(encode).hexdigest(
    #             #        ), agent.individual[0].fitness)]
    #             # n,p,f = zip(*msg)
    #             # print (i, p[:10])
    #             # ratio = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99]
    #             # for r in ratio:
    #             phenotypes = env.behavior_sampling_objects(ratio_value=0.1)
    #             # phenotypes = env.behavior_sampling_objects(method='noratio')
    #             # save the phenotype to json file
    #             phenotype_to_json(
    #                 env.pname, env.runid + '-' + str(i), phenotypes)
    #             # early_stop = validation_loop(phenotypes, 5000)
    #             validation_loop(
    #                 phenotypes, 5000, parentname=env.pname, ratio=0.1)
    #         except ValueError:
    #             pass
    #         # Plot the fitness in the graph
    #         # graph = Graph(
    #         #     env.pname, 'best.csv', [
    #         #         'explore', 'foraging', 'prospective', 'fitness'],
    #         #     pname='best' + str(i))
    #         # graph.gen_best_plots()
    #         """
    #         if early_stop:
    #             # Update the experiment table
    #             env.experiment.update_experiment()

    #             # Return phenotypes
    #             return phenotypes
    #         """
    # Update the experiment table
    foraging_percent = env.foraging_percent()
    if foraging_percent > 5:
        success = True
    else:
        success = False
    env.experiment.update_experiment_simulation(foraging_percent, success)
    """
    hashlist = dict()
    phenotypes = dict()
    # generations = [agent.individual for agent in env.agents]
    for agent in env.agents:
        encode = agent.individual[0].phenotype.encode('utf-8')
        hashval = hashlib.sha224(encode).hexdigest()
        print(
            agent.name, hashval,
            agent.individual[0].fitness, agent.food_collected)
        phenotypes[agent.individual[0].phenotype] = agent.individual[0].fitness
        try:
            hashlist[hashval] += 1
        except KeyError:
            hashlist[hashval] = 1
    print(hashlist)
    """
    # print('max, min generations', np.max(generations), np.min(generations))
    # pdb.set_trace()
    allphenotypes = env.behavior_sampling_objects(ratio_value=1.0)
    # save the phenotype to json file
    phenotype_to_json(
        env.pname, env.runid + '-' + 'all', allphenotypes)
    try:
        # return list(phenotypes.keys())
        return allphenotypes, foraging_percent, env.pname
    except UnboundLocalError:
        return None, None, None


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
    Parallel(
            n_jobs=args.threads)(delayed(learning_phase)(
                args.iter, 50, db=False) for i in range(args.runs))


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
    for n in [100, 150, 200]:
        Parallel(
            n_jobs=args.threads)(
                delayed(
                    exp_evol)(args.iter, n, args.db) for i in range(args.runs))
    # for i in range(args.runs):
    #     exp_evol_sample(args.iter, 100, args.db)


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
        '--exp_no', default=1, type=int)
    parser.add_argument('--runs', default=36, type=int)
    parser.add_argument('--threads', default=18, type=int)
    parser.add_argument('--iter', default=12000, type=int)
    parser.add_argument('--db', default=False, type=bool)
    args = parser.parse_args()
    print(args)
    experiments(args)