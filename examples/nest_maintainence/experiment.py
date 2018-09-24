"""Experiment script to run Multi source foraging simulation."""

from simmodel import SimModel
from evolmodel import EvolModel

# from swarms.utils.jsonhandler import JsonData
from swarms.utils.graph import Graph, GraphACC
from joblib import Parallel, delayed
from swarms.utils.results import SimulationResults
from swarms.utils.jsonhandler import JsonPhenotypeData
# Global variables for width and height
width = 100
height = 100

UI = False


def extract_phenotype(agents, filename, method='ratio'):
    """Extract phenotype of the learning agents.

    Sort the agents based on the overall fitness and then based on the
    method extract phenotype of the agents.
    Method can take {'ratio','higest','sample'}
    """
    sorted_agents = sorted(
        agents, key=lambda x: x.individual[0].fitness, reverse=True)

    if method == 'ratio':
        ratio_value = 0.4
        upper_bound = ratio_value * len(agents)
        selected_agents = agents[0:int(upper_bound)]
        selected_phenotype = [
            agent.individual[0].phenotype for agent in selected_agents]
        # return selected_phenotype
    else:
        selected_phenotype = [sorted_agents[0].individual[0].phenotype]
        # return [sorted_agents[0].individual[0].phenotype]

    # Save the phenotype to a json file
    JsonPhenotypeData.to_json(selected_phenotype, filename)

    # Return the phenotype
    return selected_phenotype


def after_simulation(sim, phenotypes, iteration, threshold):
    """Step for all simulation after defining environment."""
    # for all agents store the information about hub
    for agent in sim.agents:
        agent.shared_content['Hub'] = {sim.hub}
        # agent.shared_content['Sites'] = {sim.site}

    simresults = SimulationResults(
        sim.pname, sim.connect, sim.sn, sim.stepcnt,
        len(sim.debris_cleaned()), phenotypes[0]
        )

    simresults.save_phenotype()
    simresults.save_to_file()

    # Iterate and execute each step in the environment
    for i in range(iteration):
        # For every iteration we need to store the results
        # Save them into db or a file
        sim.step()
        simresults = SimulationResults(
            sim.pname, sim.connect, sim.sn, sim.stepcnt,
            len(sim.debris_cleaned()), phenotypes[0]
            )
        simresults.save_to_file()

    debris_objects = sim.debris_cleaned()

    value = len(debris_objects)

    cleaned_percent = (
        value * 100.0) / (sim.num_agents * 2.0)

    sucess = False
    print('Cleaning percent', value)

    if cleaned_percent >= threshold:
        print('Debris cleane success')
        sucess = True

    sim.experiment.update_experiment_simulation(value, sucess)

    # Plot the fitness in the graph
    graph = GraphACC(sim.pname, 'simulation.csv')
    graph.gen_plot()


def simulate_res1(env, iteration):
    """Test the performane of evolved behavior with type 1 resilience."""
    phenotypes = env[0]
    threshold = 1.0

    sim = SimModel(
        100, 100, 100, 10, iter=iteration, xmlstrings=phenotypes, pname=env[1],
        expname='NMRes1', agent='SimAgentRes1')
    sim.build_environment_from_json()

    after_simulation(sim, phenotypes, iteration, threshold)


def simulate_res2(env, iteration):
    """Test the performane of evolved behavior with type 2 resilience."""
    phenotypes = env[0]
    threshold = 1.0

    sim = SimModel(
        100, 100, 100, 10, iter=iteration, xmlstrings=phenotypes, pname=env[1],
        expname='NMRes2', agent='SimAgentRes2')
    sim.build_environment_from_json()

    after_simulation(sim, phenotypes, iteration, threshold)


def simulate(env, iteration):
    """Test the performane of evolved behavior."""
    # phenotype = agent.individual[0].phenotype
    # phenotypes = extract_phenotype(agents)
    phenotypes = env[0]
    threshold = 1.0

    sim = SimModel(
        100, 100, 100, 10, iter=iteration, xmlstrings=phenotypes, pname=env[1])
    sim.build_environment_from_json()

    after_simulation(sim, phenotypes, iteration, threshold)


def evolve(iteration):
    """Learning Algorithm block."""
    # iteration = 10000

    env = EvolModel(100, 100, 100, 10, iter=iteration)
    env.build_environment_from_json()

    # for all agents store the information about hub
    for agent in env.agents:
        agent.shared_content['Hub'] = {env.hub}

    # Iterate and execute each step in the environment
    for i in range(iteration):
        env.step()

    env.experiment.update_experiment()

    # Collecting phenotypes on the basis of debris collected
    # Find if debris has been cleaned from the hub
    debris_objects = env.debris_cleaned()

    env.phenotypes = []
    for debris in debris_objects:
        print(debris.phenotype)
        env.phenotypes += list(debris.phenotype.values())

    jfilename = env.pname + '/' + env.runid + '.json'
    JsonPhenotypeData.to_json(env.phenotypes, jfilename)

    # Not using this method right now
    # env.phenotypes = extract_phenotype(env.agents, jfilename)

    # Plot the fitness in the graph
    graph = Graph(env.pname, 'best.csv', ['explore', 'foraging'])
    graph.gen_best_plots()

    # Test the evolved behavior
    return env


def main(iter):
    """Block for the main function."""
    print('=======Start=========')
    env = evolve(iter)
    # simulate(None, iter)
    if len(env.phenotypes) > 1:
        steps = [5000 for i in range(8)]
        env = (env.phenotypes, env.pname)
        Parallel(n_jobs=4)(delayed(simulate)(env, i) for i in steps)
        Parallel(n_jobs=4)(delayed(simulate_res1)(env, i) for i in steps)
        Parallel(n_jobs=4)(delayed(simulate_res2)(env, i) for i in steps)
        # simulate(env, 10000)
    print('=======End=========')


if __name__ == '__main__':
    # Running 50 experiments in parallel
    # steps = [100000 for i in range(50)]
    # Parallel(n_jobs=8)(delayed(main)(i) for i in steps)
    # Parallel(n_jobs=4)(delayed(main)(i) for i in range(1000, 100000, 2000))
    main(900)
