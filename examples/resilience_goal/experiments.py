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
        site=None, trap=5, obs=5, width=100, height=100, notrap=1, noobs=1, nosite=1, grid=10):
    """Test the performane of evolved behavior."""
    phenotypes = env[0]
    threshold = 1.0
    # print(iteration, phenotypes)
    sim = SimForgModel(
        N, width, height, grid, iter=iteration, xmlstrings=phenotypes, pname=env[1],
        viewer=False, agent=agent, expsite=site, trap=trap, obs=obs, notrap=notrap, noobs=noobs,
        nosite=nosite)
    sim.build_environment_from_json()

    # # for all agents store the information about hub
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
    # print('trap', sim.traps[0].location, sim.traps[0].radius, sim.grid.find_grid(sim.traps[0].location))
    # print('obs', sim.obstacles[0].location, sim.obstacles[0].radius, sim.grid.find_grid(sim.obstacles[0].location))
    for i in range(iteration):
        # For every iteration we need to store the results
        # Save them into db or a file
        # print(i, [(a.name, a.location, sim.grid.find_grid(a.location), a.dead, round(a.direction, 2)) for a in sim.agents])
        sim.step()
        # print(i, [(a.name, a.location, sim.grid.find_grid(a.location), a.dead, round(a.direction, 2)) for a in sim.agents])
        # print("total dead agents",i, sim.no_agent_dead())

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
    notraps = range(1, 6)
    nosites = range(1, 5)
    trap = trap_size
    obs = obs_size
    windowsizes = [100, 200, 300, 400, 500, 600]
    gridsizes = [2, 5, 10]
    width = args.width
    height = args.height
    exp_no = args.exp_no
    no_trap = args.no_trap
    no_obs = args.no_obs
    no_site = args.no_site
    grid = args.grid
    def exp(n, agent, site, trap, obs, width, height, no_trap, no_obs, exp_no, grid, no_site):
        agent = ExecutingAgent if agent == 0 else ExecutingAgent
        dname = os.path.join(
            '/tmp', 'swarm', 'data', 'experiments', str(n), agent.__name__, str(exp_no),
            str(site), str(trap)+'_'+str(obs), str(no_trap)+'_'+str(no_obs), str(width) +'_'+str(height), str(no_site), str(grid) )
        pathlib.Path(dname).mkdir(parents=True, exist_ok=True)
        steps = [args.steps for i in range(args.runs)]
        jname = args.json_file
        phenotype = JsonPhenotypeData.load_json_file(jname)['phenotypes']
        env = (phenotype, dname)
        Parallel(
            n_jobs=args.thread)(delayed(simulate_forg)(
                env, i, agent=agent, N=n, site=site,
                trap=trap, obs=obs, width=width, height=height, notrap=no_trap, noobs=no_obs, grid=grid, nosite=no_site) for i in steps)
        # simulate_forg(env, 500, agent=agent, N=n, site=site)

    if args.all:
        for w in range(len(windowsizes)):
            for t in range(len(trapsizes)):
                for nt in notraps:
                    for site in sitelocation:
                        for agent in [0]:
                                for n in [50, 100, 200, 300, 400, 500]:
                                    for sn in [1, 2, 3, 4, 5]:
                                        pprint(n, agent, site, trapsizes[t], obssizes[t], windowsizes[w], windowsizes[w], nt, nt, 99, 10, sn)
                                    if not args.dry_run:
                                        exp(n, agent, site, trapsizes[t], obssizes[t], windowsizes[w], windowsizes[w], nt, nt, 99, 10, sn)
    else:
        if args.exp_no == 0:
            # Every thing constant just change in agent size
            for n in [50, 100, 200, 300, 400, 500]:
                pprint(n, agent, site, trap, obs, width, height, no_trap, no_obs, exp_no, grid, no_site)
                if not args.dry_run:
                    exp(n, agent, site, trap, obs, width, height, no_trap, no_obs, exp_no, grid, no_site)
        elif args.exp_no == 1:
            # Every thing constant site distance changes
            for site in sitelocation:
                pprint(n, agent, site, trap, obs, width, height, no_trap, no_obs, exp_no, grid, no_site)
                if not args.dry_run:
                    exp(n, agent, site, trap, obs, width, height, no_trap, no_obs, exp_no, grid, no_site)
        elif args.exp_no == 2:
            # Every thing constant trap/obstacle size changes
            for i in range(len(trapsizes)):
                pprint(n, agent, site, trapsizes[i], obssizes[i], width, height, no_trap, no_obs, exp_no, grid, no_site)
                if not args.dry_run:
                    exp(n, agent, site, trapsizes[i], obssizes[i],width, height, no_trap, no_obs, exp_no, grid, no_site)
        elif args.exp_no == 3:
            for w in range(len(windowsizes)):
                pprint(n, agent, site, trap, obs, windowsizes[w], windowsizes[w], no_trap, no_obs, exp_no, grid, no_site)
                if not args.dry_run:
                    exp(n, agent, site, trap, obs, windowsizes[w], windowsizes[w], no_trap, no_obs, exp_no, grid, no_site)
        elif args.exp_no == 4:
            for nt in range(1, 6):
                pprint(n, agent, site, trap, obs, width, height, nt, nt, exp_no, grid, no_site)
                if not args.dry_run:
                    exp(n, agent, site, trap, obs, width, height, nt, nt, exp_no, grid, no_site)
        elif args.exp_no == 5:
            for i in range(len(nosites)):
                pprint(n, agent, site, trap, obs, width, height, no_trap, no_obs, exp_no, grid, nosites[i])
                if not args.dry_run:
                    exp(n, agent, site, trap, obs,width, height, no_trap, no_obs, exp_no, grid, nosites[i])
        elif args.exp_no == 6:
            for grid in gridsizes:
                pprint(n, agent, site, trap, obs, width, height, no_trap, no_obs, exp_no, grid, no_site)
                if not args.dry_run:
                    exp(n, agent, site, trap, obs, width, height, no_trap, no_obs, exp_no, grid, no_site)
        else:
            pprint(n, agent, site, trap, obs, width, height, no_trap, no_obs, exp_no, grid, no_site)
            if not args.dry_run:
                exp(n, agent, site, trap, obs, width, height, no_trap, no_obs, exp_no, grid, no_site)


def pprint(n, agent, site, trap, obs, width, height, no_trap, no_obs, exp_no, grid, no_site):
    print(
        "N: %i, Site: %i, Trap: %i, Obstacles:%i, Width: %i, Height: %i, NoTrap: %i, NoObs: %i, ExpNo: %i, Grid: %i, NoSite: %i" %
        (n, site, trap, obs, width, height, no_trap, no_obs, exp_no, grid, no_site)
        )


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
    parser.add_argument('--thread', default=8, type=int)
    parser.add_argument('--runs', default=50, type=int)
    parser.add_argument('--site', default=25, type=int)
    parser.add_argument('--trap_size', default=5, type=int)
    parser.add_argument('--steps', default=500, type=int)
    parser.add_argument('--obstacle_size', default=5, type=int)
    parser.add_argument('--no_trap', default=1, type=int)
    parser.add_argument('--no_obs', default=1, type=int)
    parser.add_argument('--no_site', default=1, type=int)
    parser.add_argument('--width', default=100, type=int)
    parser.add_argument('--height', default=100, type=int)
    parser.add_argument('--dry_run', action='store_false')
    parser.add_argument('--exp_no', default=0, type=int)
    parser.add_argument('--grid', default=10, type=int)
    parser.add_argument('--json_file', default='/tmp/16321268747516-1499.json', type=str)
    parser.add_argument('--all', default=False)
    args = parser.parse_args()
    print(args)
    main(args)