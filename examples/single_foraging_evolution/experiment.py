"""Experiment script to run Single source foraging simulation."""

from model import EvolveModel, ValidationModel, TestModel
from swarms.utils.jsonhandler import JsonPhenotypeData
from swarms.utils.graph import Graph, GraphACC  # noqa : F401
from joblib import Parallel, delayed    # noqa : F401
from swarms.utils.results import SimulationResults
import py_trees
# Global variables for width and height
width = 100
height = 100

UI = False


def validation_loop(phenotypes, iteration, threshold=10.0):
    """Validate the evolved behaviors."""
    # Create a validation environment instance
    print('len of phenotype', len(set(phenotypes)))
    valid = ValidationModel(
        100, 100, 100, 10, iter=iteration)
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
            valid.foraging_percent(), phenotypes[0]
        )
        validresults.save_to_file()

    # print('food in the hub', valid.agents[0].get_food_in_hub(False))
    # Save the phenotype to json file
    phenotype_to_json(valid.pname, valid.runid + '-' + str(i), phenotypes)

    # Plot the result in the graph
    graph = GraphACC(valid.pname, 'simulation.csv')
    graph.gen_plot()

    # Return true if the sample behavior achieves a threshold
    if valid.foraging_percent() > threshold:
        return True
    else:
        return False


def test_loop(phenotypes, iteration):
    """Test the phenotypes in a completely different environment."""
    # Create a validation environment instance
    test = TestModel(
        100, 100, 100, 10, iter=iteration)
    # Build the environment
    test.build_environment_from_json()
    # Create the agents in the environment from the sampled behaviors
    test.create_agents(phenotypes=phenotypes)
    # Store the initial result
    testresults = SimulationResults(
        test.pname, test.connect, test.sn, test.stepcnt,
        test.foraging_percent(), phenotypes[0]
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

        testresults = SimulationResults(
            test.pname, test.connect, test.sn, test.stepcnt,
            test.foraging_percent(), phenotypes[0]
        )
        testresults.save_to_file()

    # Plot the result in the graph
    graph = GraphACC(test.pname, 'simulation.csv')
    graph.gen_plot()


def learning_phase(iteration, early_stop=False):
    """Learning Algorithm block."""
    # Evolution environment
    env = EvolveModel(100, 100, 100, 10, iter=iteration)
    env.build_environment_from_json()
    env.create_agents()
    # Validation Step parameter
    # Run the validation test every these many steps
    validation_step = 1000

    # Iterate and execute each step in the environment
    # Take a step i number of step in evolution environment
    # Take a 1000 step in validation environment sampling from the evolution
    # Make the validation envronmnet same as the evolution environment
    for i in range(iteration):
        # Take a step in evolution
        env.step()
        if (i + 1) % validation_step == 0:
            phenotypes = env.behavior_sampling()
            # save the phenotype to json file
            phenotype_to_json(env.pname, env.runid + '-' + str(i), phenotypes)
            early_stop = validation_loop(phenotypes, 2000)

            # Plot the fitness in the graph
            graph = Graph(
                env.pname, 'best.csv', [
                    'explore', 'foraging', 'prospective', 'fitness'],
                pname='best' + str(i))
            graph.gen_best_plots()
            """
            if early_stop:
                # Update the experiment table
                env.experiment.update_experiment()

                # Return phenotypes
                return phenotypes
            """
    # Update the experiment table
    env.experiment.update_experiment()
    return phenotypes


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
    test_loop(phenotypes, 2000)


def test_json_phenotype(json):
    # jname = '/home/aadeshnpn/Documents/BYU/hcmi/swarm/results/1543189989736115-62000EvoSForge/' + json  # noqa : E501
    jname = '/tmp/1543367322976111-8000EvoSForge/' + json
    phenotype = JsonPhenotypeData.load_json_file(jname)['phenotypes']
    # print (phenotype)
    # phenotype = ' '
    if validation_loop(phenotype, 2000):
        print('foraging success')

if __name__ == '__main__':
    # Running 50 experiments in parallel
    # Parallel(n_jobs=8)(delayed(main)(i) for i in range(2000, 100000, 2000))
    # Parallel(n_jobs=4)(delayed(main)(i) for i in range(1000, 8000, 2000))
    # main(8000)
    # json = '1543367322976111-5999.json'
    # test_json_phenotype(json)
    Parallel(n_jobs=8)(delayed(main)(8000) for i in range(16))
