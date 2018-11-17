"""Experiment script to run Single source foraging simulation."""

from model import EvolveModel, ValidationModel, TestModel
# from swarms.utils.jsonhandler import JsonData
from swarms.utils.graph import Graph, GraphACC  # noqa : F401
from joblib import Parallel, delayed    # noqa : F401
from swarms.utils.results import SimulationResults

# Global variables for width and height
width = 100
height = 100

UI = False


def validation_loop(phenotypes, iteration, threshold=50.0):
    """Validate the evolved behaviors."""
    # Create a validation environment instance
    valid = ValidationModel(
        100, 100, 100, 10, iter=iteration)
    # Build the environment
    valid.build_environment_from_json()
    # Create the agents in the environment from the sampled behaviors
    valid.create_agents(phenotypes=phenotypes)
    for i in range(iteration):
        valid.step()

    # Return true if the sample behavior achieves a threshold
    if valid.foraging_percent() > threshold:
        return True
    else:
        return False


def test_loop(phenotypes, iteration):
    """Validate the evolved behaviors."""
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


def learning_phase(iteration, early_stop=True):
    """Learning Algorithm block."""
    # Evolution environment
    env = EvolveModel(50, 100, 100, 10, iter=iteration)
    env.build_environment_from_json()

    # Validation Step parameter
    # Run the validation test every these many steps
    validation_step = 2

    # for all agents store the information about hub
    for agent in env.agents:
        agent.shared_content['Hub'] = {env.hub}

    # Iterate and execute each step in the environment
    # Take a step i number of step in evolution environment
    # Take a 1000 step in validation environment sampling from the evolution
    # Make the validation envronmnet same as the evolution environment
    for i in range(iteration):
        # Take a step in evolution
        env.step()
        if i % validation_step == 0:
            phenotypes = env.behavior_sampling()
            early_stop = validation_loop(phenotypes, 1000)
            if early_stop:
                # Save the phenotypes to a json file

                # Update the experiment table
                env.experiment.update_experiment()

                # Return phenotypes
                return phenotypes

    # Update the experiment table
    env.experiment.update_experiment()
    return phenotypes


def test_phase(phenotypes):
    """Test the phenotypes in a completely different environment."""
    pass


def main(iter):
    """Block for the main function."""
    # Run the evolutionary learning algorithm
    phenotypes = learning_phase(iter)
    # Run the evolved behaviors on a test environment
    test_phase(phenotypes)


if __name__ == '__main__':
    # Running 50 experiments in parallel

    # Parallel(n_jobs=8)(delayed(main)(i) for i in range(1000, 100000, 2000))
    # Parallel(n_jobs=4)(delayed(main)(i) for i in range(1000, 8000, 2000))
    main(1800)
