from model import SimForgModelComm

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
        site=None, trap=5, obs=5, width=100, height=100, notrap=1, noobs=1,
        signal=False, pheromone=False, action=False, condition=False):
    """Test the performane of evolved behavior."""
    phenotypes = env[0]
    threshold = 1.0

    sim = SimForgModelComm(
        N, width, height, 10, iter=iteration, xmlstrings=phenotypes, pname=env[1],
        viewer=False, agent=agent, expsite=site, trap=trap, obs=obs,
        notrap=notrap, noobs=noobs,
        signal=signal, pheromone=pheromone, action=action, condition=condition)
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


def pprint(n, width, height, signal, pheromone, action, condition, exp_no):
    print(
        "N: %i, Width: %i, Height: %i, Signal: %i, Pheromone: %i, Action: %i, Condition: %i, ExpNo: %i" %
        (n, width, height, signal, pheromone, action, condition, exp_no)
        )


def main(args):
    n = args.n
    runs = args.runs
    windowsizes = [100, 200, 300, 400, 500, 600]
    width = args.width
    height = args.height
    exp_no = args.exp_no
    signal = args.signal
    pheromone = args.pheromone
    action = args.action
    condition = args.condition
    site = 30
    def exp(n, width, height, signal, pheromone, action, condition, exp_no):
        dname = os.path.join(
            '/tmp', 'swarm', 'data', 'experiments', str(n), agent.__name__, str(exp_no),
            str(site), str(width) +'_'+str(height),
            str(signal)+'_'+str(pheromone)+'_'+str(action)+'_'+str(condition))
        pathlib.Path(dname).mkdir(parents=True, exist_ok=True)
        steps = [5000 for i in range(args.runs)]
        jname = args.json_file
        phenotype = JsonPhenotypeData.load_json_file(jname)['phenotypes'][:4]
        env = (phenotype, dname)
        Parallel(
            n_jobs=args.thread)(delayed(simulate_forg)(
                env, i, agent=ExecutingAgent, N=n, width=width, height=height, signal=signal,
                pheromone=pheromone, action=action, condition=condition, exp_no=exp_no) for i in steps)

    # simulate_forg(env, 500, agent=agent, N=n, site=site)
    # Signal Pheromone Condition Action
    # 1 False True   True True
    # 2 False True   True False
    # 3 False True    False False
    # 4 True False    True True
    # 5 True False    True False
    # 6 True False    False True
    # 7 True True     False True
    # 8 True True     True True
    # 9 True True     True False
    if args.all:
        comm_args = [
                    [False, True, True, True],
                    [False, True, True, False],
                    [False, True, False, False],
                    [True, False, True, True],
                    [True, False, True, False],
                    [True, False, False, True],
                    [True, True, False, True],
                    [True, True, True, True],
                    [True, True, True, False]
                    ]
        for i in range(len(comm_args)):
            pprint(n, width, height, signal=comm_args[i][0], pheromone=comm_args[i][1], action=comm_args[i][2], condition=comm_args[i][3], exp_no=i+1)
            if not args.dry_run:
                exp(n, width, height, signal=comm_args[i][0], pheromone=comm_args[i][1], action=comm_args[i][2], condition=comm_args[i][3], exp_no=i+1)

    else:
        if args.exp_no == 1:
            pprint(n, width, height, signal=False, pheromone=True, action=True, condition=True, exp_no=1)
            if not args.dry_run:
                exp(n, width, height, signal=False, pheromone=True, action=True, condition=True, exp_no=1)
        elif args.exp_no == 2:
            pprint(n, width, height, signal=False, pheromone=True, action=True, condition=False, exp_no=2)
            if not args.dry_run:
                exp(n, width, height, signal=False, pheromone=True, action=True, condition=False, exp_no=2)
        elif args.exp_no == 3:
            pprint(n, width, height, signal=False, pheromone=True, action=False, condition=False, exp_no=3)
            if not args.dry_run:
                exp(n, width, height, signal=False, pheromone=True, action=False, condition=False, exp_no=3)
        elif args.exp_no == 4:
            pprint(n, width, height, signal=True, pheromone=False, action=True, condition=True, exp_no=4)
            if not args.dry_run:
                exp(n, width, height, signal=True, pheromone=False, action=True, condition=True, exp_no=4)
        elif args.exp_no == 5:
            pprint(n, width, height, signal=True, pheromone=False, action=True, condition=False, exp_no=5)
            if not args.dry_run:
                exp(n, width, height, signal=True, pheromone=False, action=True, condition=False, exp_no=5)
        elif args.exp_no == 6:
            pprint(n, width, height, signal=True, pheromone=False, action=False, condition=True, exp_no=6)
            if not args.dry_run:
                exp(n, width, height, signal=True, pheromone=False, action=False, condition=True, exp_no=6)
        elif args.exp_no == 7:
            pprint(n, width, height, signal=True, pheromone=True, action=False, condition=True, exp_no=7)
            if not args.dry_run:
                exp(n, width, height, signal=True, pheromone=True, action=False, condition=True, exp_no=7)
        elif args.exp_no == 8:
            pprint(n, width, height, signal=True, pheromone=True, action=True, condition=True, exp_no=8)
            if not args.dry_run:
                exp(n, width, height, signal=True, pheromone=True, action=True, condition=True, exp_no=8)
        elif args.exp_no == 9:
            pprint(n, width, height, signal=True, pheromone=True, action=True, condition=False, exp_no=9)
            if not args.dry_run:
                exp(n, width, height, signal=True, pheromone=True, action=True, condition=False, exp_no=9)
        else:
            pprint(n, width, height, signal=False, pheromone=True, action=True, condition=True,  exp_no=99)
            if not args.dry_run:
                exp(n, width, height, signal=False, pheromone=False, action=False, condition=False,  exp_no=99)


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
    parser.add_argument('--obstacle_size', default=5, type=int)
    parser.add_argument('--no_trap', default=1, type=int)
    parser.add_argument('--no_obs', default=1, type=int)
    parser.add_argument('--width', default=100, type=int)
    parser.add_argument('--height', default=100, type=int)
    parser.add_argument('--dry_run', action='store_false')
    parser.add_argument('--action', action='store_true')
    parser.add_argument('--condition', action='store_true')
    parser.add_argument('--signal', action='store_true')
    parser.add_argument('--pheromone', action='store_true')
    parser.add_argument('--exp_no', default=1, type=int)
    parser.add_argument('--json_file', default='/tmp/16244215326204-10999.json', type=str)
    parser.add_argument('--all', action='store_true')
    args = parser.parse_args()
    print(args)
    main(args)