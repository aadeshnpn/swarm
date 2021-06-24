"""Experiment script to run Single source foraging simulation."""

import numpy as np
# import pdb
# import hashlib
import sys
from model import EvolveModel, ValidationModel, TestModel
from swarms.utils.jsonhandler import JsonPhenotypeData
from swarms.utils.graph import Graph, GraphACC  # noqa : F401
from joblib import Parallel, delayed    # noqa : F401
from swarms.utils.results import SimulationResults, SimulationResultsTraps
# import py_trees
# Global variables for width and height
width = 100
height = 100

UI = False


def validation_loop(
        phenotypes, iteration, parentname=None, ratio=1, threshold=10, n=100):
    """Validate the evolved behaviors."""
    # Create a validation environment instance
    # print('len of phenotype', len(set(phenotypes)))
    valid = ValidationModel(
        n, width, height, 10, iter=iteration, parent=parentname, ratio=ratio)
    # print('parent:', parentname, ' children:', valid.runid)
    # Build the environment
    valid.build_environment_from_json()
    # Create the agents in the environment from the sampled behaviors
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
            valid.foraging_percent(), phenotypes[0], db=True
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


def test_loop(
        phenotypes, iteration, parentname=None, ratio=1, n=100, signal=False, pheromone=False):
    """Test the phenotypes in a completely different environment."""
    # Create a validation environment instance
    test = TestModel(
        n, width, height, 10, iter=iteration, parent=parentname, ratio=ratio)
    # Build the environment
    test.build_environment_from_json()
    # Create the agents in the environment from the sampled behaviors
    test.create_agents(phenotypes=phenotypes, removesignal=signal, removepheromone=pheromone)
    # Store the initial result
    testresults = SimulationResultsTraps(
        test.pname, test.connect, test.sn, test.stepcnt,
        test.foraging_percent(), phenotypes[0], test.no_agent_dead(), db=False
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
            test.foraging_percent(), phenotypes[0], test.no_agent_dead(),db=False
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


def learning_phase(iteration, early_stop=False):
    """Learning Algorithm block."""
    # Evolution environment
    env = EvolveModel(100, width, height, 10, iter=iteration)
    env.build_environment_from_json()
    env.create_agents()
    # Validation Step parameter
    # Run the validation test every these many steps
    validation_step = 11000

    # Iterate and execute each step in the environment
    # Take a step i number of step in evolution environment
    # Take a 1000 step in validation environment sampling from the evolution
    # Make the validation envronmnet same as the evolution environment
    for i in range(iteration):
        # Take a step in evolution
        env.step()
        if (i + 1) % validation_step == 0:
            try:
                # print([agent.individual[0].fitness for agent in env.agents])
                # msg = []
                # for agent in env.agents:
                #    encode = agent.individual[0].phenotype.encode('utf-8')
                #    msg += [(
                #        agent.name, hashlib.sha224(encode).hexdigest(
                #        ), agent.individual[0].fitness)]
                # n,p,f = zip(*msg)
                # print (i, p[:10])
                # ratio = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99]
                # for r in ratio:
                phenotypes = env.behavior_sampling_objects(ratio_value=0.1)
                # phenotypes = env.behavior_sampling_objects(method='noratio')
                # save the phenotype to json file
                phenotype_to_json(
                    env.pname, env.runid + '-' + str(i), phenotypes)
                # early_stop = validation_loop(phenotypes, 5000)
                validation_loop(
                    phenotypes, 5000, parentname=env.pname, ratio=0.1)
            except ValueError:
                pass
            # Plot the fitness in the graph
            # graph = Graph(
            #     env.pname, 'best.csv', [
            #         'explore', 'foraging', 'prospective', 'fitness'],
            #     pname='best' + str(i))
            # graph.gen_best_plots()
            """
            if early_stop:
                # Update the experiment table
                env.experiment.update_experiment()

                # Return phenotypes
                return phenotypes
            """
    # Update the experiment table
    if env.foraging_percent() > 5:
        success = True
    else:
        success = False
    env.experiment.update_experiment_simulation(env.foraging_percent(), success)
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
        return phenotypes
    except UnboundLocalError:
        return None


def phenotype_to_json(pname, runid, phenotypes):
    """Store the phenotype to a json file."""
    jfilename = pname + '/' + runid + '.json'
    JsonPhenotypeData.to_json(phenotypes, jfilename)


def main(iter):
    """Block for the main function."""
    # Run the evolutionary learning algorithm
    phenotypes = learning_phase(iter)
    # learning_phase(iter)
    # Run the evolved behaviors on a test environment
    if phenotypes is not None:
        test_loop(phenotypes, 5000)


def test_json_phenotype(json):
    # 16237197451679-10999.json
    jname = '/tmp/16237201059243-all.json'# noqa : E501
    # jname = '/tmp/1543367322976111-8000EvoSForge/' + json
    phenotype = JsonPhenotypeData.load_json_file(jname)['phenotypes']
    print(len(phenotype))
    # phenotype = ' '

    # test_loop(phenotype, 5000)
    for n in [50, 100, 200, 300, 400, 500]:
        Parallel(
            n_jobs=8)(delayed(test_loop)(
                phenotype, 5000, None, n=n) for i in range(128))

    # validation_loop(phenotype, 5000, '/tmp/swarm/data/experiments/')
    # if validation_loop(phenotype, 2000, 1):
    #    print('foraging success')


def test_json_blocked_comm_behavior(json, signal=True, pheromone=True):
    jname = '/tmp/16237201059243-all.json'# noqa : E501
    # jname = '/tmp/1543367322976111-8000EvoSForge/' + json
    # jname = '/tmp/16244215326204-10999.json' # Awesome behaviors with 11 length
    phenotype = JsonPhenotypeData.load_json_file(jname)['phenotypes']
    # print(len(phenotype))
    # phenotype = ' '
    # test_loop(phenotype, 5000, None, n=5, signal=signal, pheromone=pheromone)

    # test_loop(phenotype, 5000)
    # for n in [50, 100, 200, 300, 400, 500]:
    for n in [50, 100]:# , 200, 300, 400, 500]:
        Parallel(
            n_jobs=8)(delayed(test_loop)(
                phenotype, 5000, None, n=n, signal=signal, pheromone=pheromone) for i in range(8))


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


if __name__ == '__main__':
    # Running 50 experiments in parallel
    # Parallel(n_jobs=8)(delayed(main)(i) for i in range(2000, 100000, 2000))
    # Parallel(n_jobs=4)(delayed(main)(i) for i in range(1000, 8000, 2000))
    # main(12000)
    # json = '1550083569946511-all.json'
    # test_json_phenotype(None)
    test_json_blocked_comm_behavior(None)

    # Parallel(n_jobs=4)(delayed(main)(12000) for i in range(8))
    # main(12000)
    # test_all_phenotype('/tmp/links.txt')
    # jsonlist = sys.argv
    # print ('jsonlist',len(jsonlist))
    # test_top_phenotype(jsonlist[1:])