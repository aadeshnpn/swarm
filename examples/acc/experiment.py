"""Experiment script to run Single source foraging simulation."""

from simmodel import SimModel, SimModelRes1, SimModelRes2
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


def simulate_res1(env, iteration, prob=0.5):
    """Test the performane of evolved behavior."""
    phenotypes = env[0]
    threshold = 1.0

    sim = SimModelRes1(
        150, 100, 100, 10, iter=iteration, xmlstrings=phenotypes, pname=env[1], prob=prob)
    sim.build_environment_from_json()

    # for all agents store the information about hub
    for agent in sim.agents:
        agent.shared_content['Hub'] = {sim.hub}
        # agent.shared_content['Sites'] = {sim.site}

    simresults = SimulationResults(
        sim.pname, sim.connect, sim.sn, sim.stepcnt, sim.food_in_hub(),
        phenotypes[0]
        )

    simresults.save_phenotype()
    simresults.save_to_file()

    # Iterate and execute each step in the environment
    for i in range(iteration):
        # For every iteration we need to store the results
        # Save them into db or a file
        sim.step()
        value = sim.food_in_hub()

        foraging_percent = (
            value * 100.0) / (sim.num_agents * 1.0)
        simresults = SimulationResults(
            sim.pname, sim.connect, sim.sn, sim.stepcnt, foraging_percent,
            phenotypes[0]
            )
        simresults.save_to_file()

    # print ('food at site', len(sim.food_in_loc(sim.site.location)))
    # print ('food at hub', len(sim.food_in_loc(sim.hub.location)))
    # print("Total food in the hub", len(food_objects))

    # food_objects = sim.food_in_loc(sim.hub.location)

    # for food in food_objects:
    #     print('simulate phenotye:', dir(food))

    sucess = False
    print('Foraging percent', value)

    if foraging_percent >= threshold:
        print('Foraging success')
        sucess = True

    sim.experiment.update_experiment_simulation(value, sucess)

    # Plot the fitness in the graph
    graph = GraphACC(sim.pname, 'simulation.csv')
    graph.gen_plot()


def simulate_res2(env, iteration, prob=0.5):
    """Test the performane of evolved behavior."""
    phenotypes = env[0]
    threshold = 1.0

    sim = SimModelRes2(
        150, 100, 100, 10, iter=iteration, xmlstrings=phenotypes, pname=env[1], prob=prob)
    sim.build_environment_from_json()

    # for all agents store the information about hub
    for agent in sim.agents:
        agent.shared_content['Hub'] = {sim.hub}
        # agent.shared_content['Sites'] = {sim.site}

    simresults = SimulationResults(
        sim.pname, sim.connect, sim.sn, sim.stepcnt, sim.food_in_hub(),
        phenotypes[0]
        )

    simresults.save_phenotype()
    simresults.save_to_file()

    # Iterate and execute each step in the environment
    for i in range(iteration):
        # For every iteration we need to store the results
        # Save them into db or a file
        sim.step()
        value = sim.food_in_hub()

        foraging_percent = (
            value * 100.0) / (sim.num_agents * 2.0)
        simresults = SimulationResults(
            sim.pname, sim.connect, sim.sn, sim.stepcnt, foraging_percent,
            phenotypes[0]
            )
        simresults.save_to_file()

    # print ('food at site', len(sim.food_in_loc(sim.site.location)))
    # print ('food at hub', len(sim.food_in_loc(sim.hub.location)))
    # print("Total food in the hub", len(food_objects))

    # food_objects = sim.food_in_loc(sim.hub.location)

    # for food in food_objects:
    #    print('simulate phenotye:', dir(food))

    sucess = False
    print('Foraging percent', value)

    if foraging_percent >= threshold:
        print('Foraging success')
        sucess = True

    sim.experiment.update_experiment_simulation(value, sucess)

    # Plot the fitness in the graph
    graph = GraphACC(sim.pname, 'simulation.csv')
    graph.gen_plot()


def simulate(env, iteration, N=100):
    """Test the performane of evolved behavior."""
    # phenotype = agent.individual[0].phenotype
    # phenotypes = extract_phenotype(agents)
    phenotypes = env[0]
    # phenotypes = ['<?xml version=\"1.0\" encoding=\"UTF-8\"?><Sequence><Sequence><Sequence><Selector><cond>IsDropable_Sites</cond><act>Explore</act></Selector><Sequence><cond>NeighbourObjects</cond><act>CompositeSingleCarry_Food</act></Sequence></Sequence> <Sequence><Sequence><cond>NeighbourObjects</cond><act>Explore</act></Sequence> <Sequence><cond>NeighbourObjects</cond><act>MoveTowards_Hub</act></Sequence></Sequence></Sequence> <Sequence><Selector><cond>NeighbourObjects</cond><act>MoveAway_Sites</act></Selector><Sequence><cond>NeighbourObjects</cond><act>CompositeDrop_Food</act></Sequence></Sequence></Sequence>', '<?xml version=\"1.0\" encoding=\"UTF-8\"?><Sequence><Sequence><Sequence><Selector><cond>IsDropable_Sites</cond><act>Explore</act></Selector><Sequence><cond>NeighbourObjects</cond><act>CompositeSingleCarry_Food</act></Sequence></Sequence> <Sequence><Sequence><cond>NeighbourObjects</cond><act>Explore</act></Sequence> <Sequence><cond>NeighbourObjects</cond><act>MoveTowards_Hub</act></Sequence></Sequence></Sequence> <Sequence><Selector><cond>NeighbourObjects</cond><act>MoveAway_Sites</act></Selector><Sequence><cond>NeighbourObjects</cond><act>CompositeDrop_Food</act></Sequence></Sequence></Sequence>','<?xml version=\"1.0\" encoding=\"UTF-8\"?><Sequence><Sequence><Sequence><Selector><cond>IsDropable_Sites</cond><act>Explore</act></Selector><Sequence><cond>NeighbourObjects</cond><act>CompositeSingleCarry_Food</act></Sequence></Sequence> <Sequence><Sequence><cond>NeighbourObjects</cond><act>Explore</act></Sequence> <Sequence><cond>NeighbourObjects</cond><act>MoveTowards_Hub</act></Sequence></Sequence></Sequence> <Sequence><Selector><cond>NeighbourObjects</cond><act>MoveAway_Sites</act></Selector><Sequence><cond>NeighbourObjects</cond><act>CompositeDrop_Food</act></Sequence></Sequence></Sequence>','<?xml version=\"1.0\" encoding=\"UTF-8\"?><Sequence><Sequence><Sequence><Selector><cond>IsDropable_Sites</cond><act>Explore</act></Selector><Sequence><cond>NeighbourObjects</cond><act>CompositeSingleCarry_Food</act></Sequence></Sequence> <Sequence><Sequence><cond>NeighbourObjects</cond><act>Explore</act></Sequence> <Sequence><cond>NeighbourObjects</cond><act>MoveTowards_Hub</act></Sequence></Sequence></Sequence> <Sequence><Selector><cond>NeighbourObjects</cond><act>MoveAway_Sites</act></Selector><Sequence><cond>NeighbourObjects</cond><act>CompositeDrop_Food</act></Sequence></Sequence></Sequence>','<?xml version=\"1.0\" encoding=\"UTF-8\"?><Sequence><Sequence><Sequence><Sequence><Sequence><cond>NeighbourObjects</cond><act>Explore</act></Sequence> <Sequence><cond>NeighbourObjects</cond><act>CompositeSingleCarry_Food</act></Sequence></Sequence> <Sequence><cond>IsDropable_Hub</cond><act>CompositeDrop_Food</act></Sequence></Sequence> <Sequence><cond>IsDropable_Hub</cond><cond>IsDropable_Hub</cond><act>CompositeDrop_Food</act></Sequence></Sequence> <Selector><Selector><cond>IsDropable_Hub</cond><act>MoveTowards_Hub</act></Selector> <Selector><cond>IsDropable_Hub</cond><cond>NeighbourObjects</cond><act>MoveTowards_Sites</act></Selector></Selector></Sequence>','<?xml version=\"1.0\" encoding=\"UTF-8\"?><Sequence><Sequence><Sequence><Sequence><Sequence><cond>NeighbourObjects</cond><act>Explore</act></Sequence> <Sequence><cond>NeighbourObjects</cond><act>CompositeSingleCarry_Food</act></Sequence></Sequence> <Sequence><cond>IsDropable_Hub</cond><act>CompositeDrop_Food</act></Sequence></Sequence> <Sequence><cond>IsDropable_Hub</cond><cond>IsDropable_Hub</cond><act>CompositeDrop_Food</act></Sequence></Sequence> <Selector><Selector><cond>IsDropable_Hub</cond><act>MoveTowards_Hub</act></Selector> <Selector><cond>IsDropable_Hub</cond><cond>NeighbourObjects</cond><act>MoveTowards_Sites</act></Selector></Selector></Sequence>','<?xml version=\"1.0\" encoding=\"UTF-8\"?><Selector><Selector><Sequence><Sequence><cond>IsDropable_Hub</cond><act>Explore</act></Sequence> <Sequence><cond>NeighbourObjects</cond><act>CompositeDrop_Food</act></Sequence></Sequence><Sequence><Sequence><cond>IsDropable_Hub</cond><act>CompositeSingleCarry_Food</act></Sequence> <Selector><cond>NeighbourObjects</cond><act>MoveTowards_Sites</act></Selector></Sequence></Selector> <Selector><Selector><cond>IsDropable_Sites</cond><cond>IsDropable_Hub</cond><act>CompositeSingleCarry_Food</act></Selector> <Selector><cond>IsDropable_Hub</cond><cond>NeighbourObjects</cond><act>CompositeDrop_Food</act></Selector></Selector></Selector>','<?xml version=\"1.0\" encoding=\"UTF-8\"?><Selector><Selector><Sequence><Sequence><cond>IsDropable_Hub</cond><act>Explore</act></Sequence> <Sequence><cond>NeighbourObjects</cond><act>CompositeDrop_Food</act></Sequence></Sequence><Sequence><Sequence><cond>IsDropable_Hub</cond><act>CompositeSingleCarry_Food</act></Sequence> <Selector><cond>NeighbourObjects</cond><act>MoveTowards_Sites</act></Selector></Sequence></Selector> <Selector><Selector><cond>IsDropable_Sites</cond><cond>IsDropable_Hub</cond><act>CompositeSingleCarry_Food</act></Selector> <Selector><cond>IsDropable_Hub</cond><cond>NeighbourObjects</cond><act>CompositeDrop_Food</act></Selector></Selector></Selector>','<?xml version=\"1.0\" encoding=\"UTF-8\"?><Selector><Selector><Sequence><Sequence><cond>IsDropable_Hub</cond><act>Explore</act></Sequence> <Sequence><cond>NeighbourObjects</cond><act>CompositeDrop_Food</act></Sequence></Sequence><Sequence><Sequence><cond>IsDropable_Hub</cond><act>CompositeSingleCarry_Food</act></Sequence> <Selector><cond>NeighbourObjects</cond><act>MoveTowards_Sites</act></Selector></Sequence></Selector> <Selector><Selector><cond>IsDropable_Sites</cond><cond>IsDropable_Hub</cond><act>CompositeSingleCarry_Food</act></Selector> <Selector><cond>IsDropable_Hub</cond><cond>NeighbourObjects</cond><act>CompositeDrop_Food</act></Selector></Selector></Selector>','<?xml version=\"1.0\" encoding=\"UTF-8\"?><Selector><Selector><Sequence><Sequence><cond>IsDropable_Hub</cond><act>Explore</act></Sequence> <Sequence><cond>NeighbourObjects</cond><act>CompositeDrop_Food</act></Sequence></Sequence><Sequence><Sequence><cond>IsDropable_Hub</cond><act>CompositeSingleCarry_Food</act></Sequence> <Selector><cond>NeighbourObjects</cond><act>MoveTowards_Sites</act></Selector></Sequence></Selector> <Selector><Selector><cond>IsDropable_Sites</cond><cond>IsDropable_Hub</cond><act>CompositeSingleCarry_Food</act></Selector> <Selector><cond>IsDropable_Hub</cond><cond>NeighbourObjects</cond><act>CompositeDrop_Food</act></Selector></Selector></Selector>']
    threshold = 1.0

    sim = SimModel(
        N, 100, 100, 10, iter=iteration, xmlstrings=phenotypes, pname=env[1])
    sim.build_environment_from_json()

    # for all agents store the information about hub
    for agent in sim.agents:
        agent.shared_content['Hub'] = {sim.hub}
        # agent.shared_content['Sites'] = {sim.site}

    simresults = SimulationResults(
        sim.pname, sim.connect, sim.sn, sim.stepcnt, sim.food_in_hub(),
        phenotypes[0]
        )

    simresults.save_phenotype()
    simresults.save_to_file()

    # Iterate and execute each step in the environment
    for i in range(iteration):
        # For every iteration we need to store the results
        # Save them into db or a file
        sim.step()
        value = sim.food_in_hub()
        foraging_percent = (
            value * 100.0) / (sim.num_agents * 1.0)

        simresults = SimulationResults(
            sim.pname, sim.connect, sim.sn, sim.stepcnt, foraging_percent,
            phenotypes[0]
            )

        simresults.save_to_file()

    # print ('food at site', len(sim.food_in_loc(sim.site.location)))
    # print ('food at hub', len(sim.food_in_loc(sim.hub.location)))
    # print("Total food in the hub", len(food_objects))

    # food_objects = sim.food_in_loc(sim.hub.location)

    # for food in food_objects:
    #    print('simulate phenotye:', dir(food))

    sucess = False
    print('Foraging percent', value)

    if foraging_percent >= threshold:
        print('Foraging success')
        sucess = True

    sim.experiment.update_experiment_simulation(value, sucess)

    # Plot the fitness in the graph
    graph = GraphACC(sim.pname, 'simulation.csv')
    graph.gen_plot()


def evolve(iteration):
    """Learning Algorithm block."""
    # iteration = 10000

    env = EvolModel(150, 100, 100, 10, iter=iteration)
    env.build_environment_from_json()

    # for all agents store the information about hub
    for agent in env.agents:
        agent.shared_content['Hub'] = {env.hub}

    # Hub and site object
    # print(env.hub, env.site)

    # Iterate and execute each step in the environment
    for i in range(iteration):
        env.step()

    env.experiment.update_experiment()
    jfilename = env.pname + '/' + env.runid + '.json'
    """
    # Collecting phenotypes on the basis of food collected
    # Find if food has been deposited in the hub
    food_objects = env.food_in_loc(env.hub.location)
    # print('Total food in the hub evolution:', len(food_objects))
    env.phenotypes = []
    for food in food_objects:
        print(food.phenotype)
        env.phenotypes += list(food.phenotype.values())

    JsonPhenotypeData.to_json(env.phenotypes, jfilename)
    """
    # Not using this method right now
    env.phenotypes = extract_phenotype(env.agents, jfilename)

    # Plot the fitness in the graph
    graph = Graph(env.pname, 'best.csv', ['explore', 'foraging'])
    graph.gen_best_plots()

    # Test the evolved behavior
    return env


def main(iter):
    """Block for the main function."""
    print('=======Start=========')
    steps = list(range(10000, 1000000, 2000))
    for j in range(0, len(steps), 16):
        envs = Parallel(n_jobs=16)(delayed(evolve)(i) for i in steps[j:j+16])
        # env = evolve(iter)
        # simulate(None, iter)
        """
        # Read from the json
        pname = '/home/aadeshnpn/Documents/BYU/hcmi/swarm/resilience/foraging'
        jfilename = pname + '/1538473090382007.json'
        jdata = JsonPhenotypeData.load_json_file(jfilename)
        phenotypes = jdata['phenotypes']
        """
        for env in envs:
            if len(env.phenotypes) > 1:
                steps = [5000 for i in range(16)]
                # env = (env.phenotypes, env.pname)
                # aname = pname + '/' + str(N)
                # env = (phenotypes, pname)
                Parallel(n_jobs=16)(delayed(simulate)(env, i, 150) for i in steps)
                # Run over different probability
                for p in [0.5, 0.6, 0.7, 0.8, 0.9, 0.99]:
                    Parallel(n_jobs=16)(delayed(simulate_res1)(env, i, prob=p) for i in steps)
                    Parallel(n_jobs=16)(delayed(simulate_res2)(env, i, prob=p) for i in steps)

    # for i in steps:
    #    simulate(env, i)

    # Parallel(n_jobs=4)(delayed(simulate_res2)(env, i) for i in steps)
    # simulate(env, 10000)
    print('=======End=========')


if __name__ == '__main__':
    # Running 50 experiments in parallel
    # steps = [100000 for i in range(50)]
    # Parallel(n_jobs=8)(delayed(main)(i) for i in steps)
    # Parallel(n_jobs=16)(delayed(main)(i) for i in range(10000, 1000000, 2000))
    main(90000)
    #for i in range(10000, 1000000, 2000):
    #    main(i)

