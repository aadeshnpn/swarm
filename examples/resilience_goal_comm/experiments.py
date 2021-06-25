def pprint(n, width, height, signal, pheromone, action, condition, exp_no):
    print(
        "N: %i, Width: %i, Height: %i, Signal: %i, Pheromone: %i, Action: %i, Condition: %i, ExpNo: %i", %
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
                env, i, agent=ExecutingAgent, N=n, site=site,
                trap=trap, obs=obs, width=width, height=height, notrap=no_trap, noobs=no_obs) for i in steps)
        # simulate_forg(env, 500, agent=agent, N=n, site=site)

    if args.all:
        for w in range(len(windowsizes)):
            for t in range(len(trapsizes)):
                for nt in notraps:
                    for site in sitelocation:
                        for agent in [0]:
                                for n in [50, 100, 200, 300, 400, 500]:
                                    pprint(n, agent, site, trapsizes[t], obssizes[t], windowsizes[w], windowsizes[w], nt, nt, 99)
                                    if not args.dry_run:
                                        exp(n, agent, site, trapsizes[t], obssizes[t], windowsizes[w], windowsizes[w], nt, nt, 99)
    else:
        if args.exp_no == 0:
            # Every thing constant just change in agent size
            for n in [50, 100, 200, 300, 400, 500]:
                pprint(n, agent, site, trap, obs, width, height, no_trap, no_obs, exp_no)
                if not args.dry_run:
                    exp(n, agent, site, trap, obs, width, height, no_trap, no_obs, exp_no)
        elif args.exp_no ==1:
            # Every thing constant site distance changes
            for site in sitelocation:
                pprint(n, agent, site, trap, obs, width, height, no_trap, no_obs, exp_no)
                if not args.dry_run:
                    exp(n, agent, site, trap, obs, width, height, no_trap, no_obs, exp_no)
        elif args.exp_no ==2:
            # Every thing constant trap/obstacle size changes
            for i in range(len(trapsizes)):
                pprint(n, agent, site, trapsizes[i], obssizes[i], width, height, no_trap, no_obs, exp_no)
                if not args.dry_run:
                    exp(n, agent, site, trapsizes[i], obssizes[i],width, height, no_trap, no_obs, exp_no)
        elif args.exp_no ==3:
            for w in range(len(windowsizes)):
                pprint(n, agent, site, trap, obs, windowsizes[w], windowsizes[w], no_trap, no_obs, exp_no)
                if not args.dry_run:
                    exp(n, agent, site, trap, obs, windowsizes[w], windowsizes[w], no_trap, no_obs, exp_no)
        elif args.exp_no==4:
            for nt in range(1, 6):
                pprint(n, agent, site, trap, obs, width, height, nt, nt, exp_no)
                if not args.dry_run:
                    exp(n, agent, site, trap, obs, width, height, nt, nt, exp_no)
        else:
            pprint(n, agent, site, trap, obs, width, height, no_trap, no_obs, exp_no)
            if not args.dry_run:
                exp(n, agent, site, trap, obs, width, height, no_trap, no_obs, exp_no)
