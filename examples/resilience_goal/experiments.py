from model import SimForgModel

# from swarms.utils.jsonhandler import JsonData
from swarms.utils.graph import GraphACC
from joblib import Parallel, delayed    # noqa : F401
from swarms.utils.results import SimulationResults, SimulationResultsTraps
from swarms.utils.jsonhandler import JsonPhenotypeData
from agent import ExecutingAgent
import argparse
import os
import pathlib
# Global variables for width and height
width = 100
height = 100

UI = False


def simulate_forg(
        env, iteration, agent=ExecutingAgent, N=100,
        site=None, trap=5, obs=5, width=100, height=100):
    """Test the performane of evolved behavior."""
    phenotypes = env[0]
    threshold = 1.0
    # print(iteration, phenotypes)
    sim = SimForgModel(
        N, width, height, 10, iter=iteration, xmlstrings=phenotypes, pname=env[1],
        viewer=False, agent=agent, expsite=site, trap=trap, obs=obs)
    sim.build_environment_from_json()

    # for all agents store the information about hub
    for agent in sim.agents:
        agent.shared_content['Hub'] = {sim.hub}
        # agent.shared_content['Sites'] = {sim.site}

    simresults = SimulationResultsTraps(
        sim.pname, sim.connect, sim.sn, sim.stepcnt, sim.food_in_hub(),
        phenotypes[0], sim.no_agent_dead()
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
            phenotypes[0], sim.no_agent_dead()
            )
        simresults.save_to_file()


def main(args):
    """Block for the main function."""
    # print('=======Start=========')
    # pname = '/home/aadeshnpn/Documents/BYU/HCMI/research/handcoded/nm'
    # pname = '/home/aadeshnpn/Documents/BYU/hcmi/hri/handcoded/ct'
    # directory = '/tmp/goal/data/experiments/' + str(args.n) + '/' + args.agent.__name__ + '/'
    n = args.n
    agent = args.agent
    runs = args.runs
    trap_size = args.trap_size
    obs_size = args.obstacle_size
    # site = sitelocation[args.site]
    sitelocation = [20] * 10 + [25] * 10 + [30] * 10 + [40] * 10 + [50]*10 + [60] *10 + [70]*10 + [80]*10 + [90]* 10
    site = sitelocation[args.site]
    trapsizes = range(5, 30, 5)
    obssizes = range(5, 30, 5)
    trap = trapsizes[0]
    obs = obssizes[0]
    windowsizes = [100, 200, 300, 400, 500, 600]
    def exp(n, agent, runs, site, trap, obs, width, height):
        agent = ExecutingAgent if agent == 0 else ExecutingAgent
        dname = os.path.join(
            '/tmp', 'swarm', 'data', 'experiments', str(n), agent.__name__,
            str(site), str(trap)+'_'+str(obs), str(width) +'_'+str(height) )
        pathlib.Path(dname).mkdir(parents=True, exist_ok=True)
        steps = [5000 for i in range(args.runs)]
        jname = '/tmp/16235340355923-10999.json'
        phenotype = JsonPhenotypeData.load_json_file(jname)['phenotypes']
        env = (phenotype, dname)
        Parallel(
            n_jobs=8)(delayed(simulate_forg)(
                env, i, agent=agent, N=n, site=site,
                trap=trap, obs=obs, width=width, height=height) for i in steps)
        # simulate_forg(env, 500, agent=agent, N=n, site=site)

    if args.all:
        for w in range(len(windowsizes)):
            for t in range(len(trapsizes)):
                for site in sitelocation:
                    for agent in [0]:
                            for n in [50, 100, 200, 300, 400, 500]:
                                exp(n, agent, runs, site, trapsizes[t], obssizes[t], windowsizes[w], windowsizes[w])
    else:
        if args.exp_no == 0:
            # Every thing constant just change in agent size
            for n in [50, 100, 200, 300, 400, 500]:
                exp(n, agent, runs, site, trap, obs)
        elif args.exp_no ==1:
            # Every thing constant site distance changes
            for site in sitelocation:
                exp(n, agent, runs, site, trap, obs)
        elif args.exp_no ==2:
            # Every thing constant trap/obstacle size changes
            for i in range(len(trapsizes)):
                exp(n, agent, runs, site, trapsizes[i], obssizes[i])
        elif args.exp_np ==3:
            for w in range(len(windowsizes)):
                exp(n, agent, runs, site, trap, obs, windowsizes[w], windowsizes[w])

        exp(n, agent, runs, site, trap, obs)



if __name__ == '__main__':
    # Running 50 experiments in parallel
    # steps = [100000 for i in range(50)]
    # Parallel(n_jobs=8)(delayed(main)(i) for i in steps)
    # Parallel(n_jobs=16)(delayed(main)(i) for i in range(1000, 100000, 2000))
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--n', default=100, type=int)
    # [SimForgAgentWith, SimForgAgentWithout])
    parser.add_argument('--agent', default=1, choices=[0, 1], type=int)
    parser.add_argument('--runs', default=20, type=int)
    parser.add_argument('--site', default=0, type=int)
    parser.add_argument('--trap_size', default=5, type=int)
    parser.add_argument('--obstacle_size', default=5, type=int)
    parser.add_argument('--exp_no', default=0, type=int)
    # parser.add_argument('--all', default=False)
    args = parser.parse_args()
    print(args)
    main(args)