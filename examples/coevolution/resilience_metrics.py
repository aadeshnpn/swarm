"""Script to compute resilience metric"""
"""for all four types of perturbations for both swarm tasks."""
import os
import numpy as np
import pathlib


def read_data_n_agent_perturbations_all(
        n=100, iter=12000, threshold=10, gstep=200, expp=2,
        addobject=None, removeobject=None, no_objects=1, radius=5,
        time=13000, iprob=0.85, idx=[2], fname='EvoCoevolutionPPA'):
    # maindir = '/tmp/swarm/data/experiments/' + fname
    nadir = os.path.join(
                '/tmp', 'swarm', 'data', 'experiments', fname,
                str(n), str(iter), str(threshold), str(gstep), str(expp),
                str(addobject), str(removeobject),
                str(no_objects), str(radius),
                str(time), str(iprob)
                )
    print(nadir)
    folders = pathlib.Path(nadir).glob("*EvoCoevolutionPPA")
    flist = []
    fdata = []
    mdata = []
    for i in range(len(idx)):
        # print(i)
        fdata.append(list())
    # print(list(folders))
    folders = list(folders)
    if len(folders) > 0:
        for f in folders:
            flist = [p for p in pathlib.Path(f).iterdir() if p.is_file() and p.match('simulation.csv')]
            if len(flist) > 0:
                data = np.genfromtxt(flist[0], autostrip=True, unpack=True, delimiter='|')
                # print(data.shape, flist[0])
                if data.shape[1] == 12002:
                    for i in range(len(idx)):
                        fdata[i].append(data[idx[i]])
                    # print(fdata)
        fdata = np.array(fdata)
        return fdata
    else:
        return None


def ablation_efficiency_power_foraging(t=7, time=1000):
    no_obstacles = [1, 2, 3, 4, 5]
    times = list(range(1000, 5001, 1000))
    baseline = read_data_n_agent_perturbations_all(
        n=100, iter=12000, threshold=t, time=10000, iprob=0.85,
        addobject='None', radius=5, idx=[2])
    powerbaseline100 = np.squeeze(baseline)[:, -1]

    effbaseline100 = np.squeeze(baseline)
    effbaseline100 = [np.squeeze(
        np.argwhere(
            effbaseline100[i, :] >= 80)) for i in range(
                effbaseline100.shape[0])]
    effbaseline100 = [np.array(
        [effbaseline100[i][0] if effbaseline100[
            i].shape[0] > 1 else 12000 for i in range(
                len(effbaseline100))])]
    effbaseline100 = [(((12000-effbaseline100[i])/12000)*100) for i in range(len(effbaseline100))]
    # print(powerbaseline100, effbaseline100)
    allpower = {0: powerbaseline100}
    allefficiency = {0: effbaseline100}
    for o in no_obstacles:
        data = [read_data_n_agent_perturbations_all(
            n=100, iter=12000, threshold=t, time=times[j], iprob=0.85,
            addobject='Obstacles', no_objects=o,
            radius=10, idx=[2]) for j in range(len(times))]
        # print(len(data))
        powerdata100 = [np.squeeze(
            data[i])[:, -1] for i in range(
                len(times)) if data[i] is not None]
        allpower[o] = np.concatenate(powerdata100, axis=0)
        # powerdata100 = [powerbaseline100] + powerdata100
        # print('power', powerdata100)
        effdata100 = []
        for t in range(len(times)):
            # print(o, t)
            if data[t] is not None:
                # print(o, data[t].shape)
                effdata = np.squeeze(data[t])
                effdata = [np.squeeze(np.argwhere(effdata[i, :]>=80)) for i in range(effdata.shape[0])]
                effdata = [np.array([effdata[i][0] if effdata[i].shape[0]>1 else 12000 for i in range(len(effdata))])]
                effdata = [(((12000-effdata[i])/12000)*100) for i in range(len(effdata))]
                effdata100 += effdata
            # print('eff', effdata100)
        allefficiency[o] = np.round(np.concatenate(effdata100, axis=0), 2)
    # print(len(allpower[1]))
    # print(allpower[1], allefficiency[1])


def main():
    ablation_efficiency_power_foraging()


if __name__ == "__main__":
    main()
