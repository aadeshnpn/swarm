"""Experiment script to run handcoded simulation."""

from simmodel import SimForgModel, EvolModel

# from swarms.utils.jsonhandler import JsonData
from swarms.utils.graph import GraphACC
from joblib import Parallel, delayed    # noqa : F401
from swarms.utils.results import SimulationResults, SimulationResultsTraps
from swarms.utils.jsonhandler import JsonPhenotypeData
from simagent import SimForgAgentWith, SimForgAgentWithout
import argparse
import os
import pathlib
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


def simulate_forg(env, iteration, agent=SimForgAgentWith, N=100, site=None):
    """Test the performane of evolved behavior."""
    phenotypes = env[0]
    threshold = 1.0

    sim = SimForgModel(
        N, 200, 200, 10, iter=iteration, xmlstrings=phenotypes, pname=env[1], viewer=False, agent=agent, expsite=site)
    sim.build_environment_from_json()

    # for all agents store the information about hub
    for agent in sim.agents:
        agent.shared_content['Hub'] = {sim.hub}
        # agent.shared_content['Sites'] = {sim.site}

    simresults = SimulationResultsTraps(
        sim.pname, sim.connect, sim.sn, sim.stepcnt, sim.food_in_hub(),
        phenotypes[0], sum([1 if a.dead else 0 for a in sim.agents])
        )

    # simresults.save_phenotype()
    simresults.save_to_file()

    # Iterate and execute each step in the environment
    for i in range(iteration):
        # For every iteration we need to store the results
        # Save them into db or a file
        sim.step()
        # print("total dead agents",i, )        
        value = sim.food_in_hub()

        foraging_percent = (
            value * 100.0) / (sim.num_agents * 1)

        simresults = SimulationResultsTraps(
            sim.pname, sim.connect, sim.sn, sim.stepcnt, foraging_percent,
            phenotypes[0], sum([1 if a.dead else 0 for a in sim.agents])
            )
        simresults.save_to_file()

    # print ('food at site', len(sim.food_in_loc(sim.site.location)))
    # print ('food at hub', len(sim.food_in_loc(sim.hub.location)))
    # print("Total food in the hub", len(food_objects))
    print("total dead agents", sum([1 if a.dead else 0 for a in sim.agents]))
    # food_objects = sim.food_in_loc(sim.hub.location)

    # for food in food_objects:
    #    print('simulate phenotye:', dir(food))

    # sucess = False
    # print('Foraging percent', value)

    # if foraging_percent >= threshold:
    #     print('Foraging success')
    #     sucess = True

    # sim.experiment.update_experiment_simulation(foraging_percent, sucess)

    # Plot the fitness in the graph
    # graph = GraphACC(sim.pname, 'simulation.csv')
    # graph.gen_plot()


def main(args):
    """Block for the main function."""
    # print('=======Start=========')
    # pname = '/home/aadeshnpn/Documents/BYU/HCMI/research/handcoded/nm'
    # pname = '/home/aadeshnpn/Documents/BYU/hcmi/hri/handcoded/ct'
    # directory = '/tmp/goal/data/experiments/' + str(args.n) + '/' + args.agent.__name__ + '/'
    n = args.n
    agent = args.agent
    runs = args.runs
    
    sitelocation  = [ 
        {"x":50, "y":-50, "radius":10, "q_value":0.9},
        {"x":50, "y":50, "radius":10, "q_value":0.9},        
        {"x":-50, "y":50, "radius":10, "q_value":0.9},                
        {"x":30, "y":-30, "radius":10, "q_value":0.9},
        {"x":30, "y":30, "radius":10, "q_value":0.9},        
        {"x":-30, "y":30, "radius":10, "q_value":0.9},                        
        {"x":90, "y":-90, "radius":10, "q_value":0.9},
        {"x":-90, "y":90, "radius":10, "q_value":0.9}, 
    ]
    site = sitelocation[args.site]

    def exp(n, agent, runs, site):
        agent = SimForgAgentWith if agent == 0 else SimForgAgentWithout
        dname = os.path.join('/tmp', 'swarm', 'data', 'experiments', str(n), agent.__name__, str(site['x'])+str(site['y']))
        pathlib.Path(dname).mkdir(parents=True, exist_ok=True)
        steps = [1500 for i in range(args.runs)]
        env = (['123', '123'], dname)
        # Parallel(n_jobs=8)(delayed(simulate_forg)(env, i, agent=agent, N=n, site=site) for i in steps)
        simulate_forg(env, 1500, agent=agent, N=n, site=site)

    if args.all:
        # for agent in [0, 1]:
        for agent in [0, 1]:        
            # for n in [50, 100, 200, 300, 400, 500]:
            # for n in [100, 200, 300, 400]:            
            for n in [100]:                        
                exp(n, agent, runs, site)
    else:
        exp(n, agent, runs, site)



if __name__ == '__main__':
    # Running 50 experiments in parallel
    # steps = [100000 for i in range(50)]
    # Parallel(n_jobs=8)(delayed(main)(i) for i in steps)
    # Parallel(n_jobs=16)(delayed(main)(i) for i in range(1000, 100000, 2000))
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--n', default=50, type=int)
    # [SimForgAgentWith, SimForgAgentWithout])        
    parser.add_argument('--agent', default=1, choices=[0, 1], type=int)
    parser.add_argument('--runs', default=20, type=int)
    parser.add_argument('--site', default=7, type=int)    
    parser.add_argument('--all', default=False)
    args = parser.parse_args()
    print(args)
    main(args)