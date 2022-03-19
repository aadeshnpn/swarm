"""Script to compute resilience metric"""
"""for all four types of perturbations for both swarm tasks."""
from distutils import dist
import os
import numpy as np
import pathlib
from sklearn.linear_model import LinearRegression
import matplotlib
# If there is $DISPLAY, display the plot
if os.name == 'posix' and "DISPLAY" not in os.environ:
    matplotlib.use('Agg')

import matplotlib.pyplot as plt     # noqa: E402


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
    # mdata = []
    for i in range(len(idx)):
        # print(i)
        fdata.append(list())
    # print(list(folders))
    folders = list(folders)
    if len(folders) > 0:
        for f in folders:
            flist = [p for p in pathlib.Path(
                f).iterdir() if p.is_file() and p.match('simulation.csv')]
            if len(flist) > 0:
                data = np.genfromtxt(
                    flist[0], autostrip=True, unpack=True, delimiter='|')
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
    times = list(range(1000, 12001, 1000))
    baseline = read_data_n_agent_perturbations_all(
        n=100, iter=12000, threshold=t, time=10000, iprob=0.85,
        addobject='None', radius=5, idx=[2])
    # powerbaseline100 = np.squeeze(baseline)[:, -1]
    effbaseline100 = np.squeeze(baseline)
    effbaseline100 = [np.squeeze(
        np.argwhere(
            effbaseline100[i, :] >= 80)) for i in range(
                effbaseline100.shape[0])]
    effbaseline100 = [np.array(
        [effbaseline100[i][0] if effbaseline100[
            i].shape[0] > 1 else 12000 for i in range(
                len(effbaseline100))])]
    effbaseline100 = [(
        ((12000-effbaseline100[i])/12000)*100) for i in range(
            len(effbaseline100))]
    # print(powerbaseline100, effbaseline100)
    # allpower = {0: [np.median(powerbaseline100)]}
    # allefficiency = {0: [np.median(effbaseline100)]}
    allpower = {}
    allefficiency = {}

    for o in no_obstacles:
        data = [read_data_n_agent_perturbations_all(
            n=100, iter=12000, threshold=t, time=times[j], iprob=0.85,
            addobject='Obstacles', no_objects=o,
            radius=10, idx=[2]) for j in range(len(times))]
        # print(len(data))
        powerdata100 = [np.squeeze(
            data[i])[:, -1] for i in range(
                len(times)) if data[i] is not None]
        # powerdata100 = [np.median(
        # powerdata100[i]) for i in range(len(powerdata100))]
        # allpower[o] = powerdata100
        allpower[o] = np.concatenate(powerdata100, axis=0)

        effdata100 = []
        for ti in range(len(times)):
            # print(o, t)
            if data[ti] is not None:
                # print(o, data[t].shape)
                effdata = np.squeeze(data[ti])
                effdata = [np.squeeze(
                    np.argwhere(
                        effdata[i, :] >= 80)) for i in range(effdata.shape[0])]
                effdata = [np.array(
                    [effdata[i][
                        0] if effdata[i].shape[
                            0] > 1 else 12000 for i in range(len(effdata))])]
                effdata = [(
                    (
                        (12000-effdata[i])/12000)*100) for i in range(
                            len(effdata))]
                # effdata = [np.median(effdata[i]
                # ) for i in range(len(effdata))]
                effdata100 += effdata
        # allefficiency[o] = np.round(effdata100, 2)
        # allefficiency[o] = np.round(np.array(effdata100), 2)
        allefficiency[o] = np.round(np.concatenate(effdata100), 2)

    print(allpower, allefficiency)
    # linear_regressor = LinearRegression()  # create object for the class
    X = np.concatenate(np.array([[k] * len(v) for (k, v) in allpower.items()]))
    Y = np.concatenate(np.array([v for _, v in allpower.items()]))
    # X = X.reshape((X.shape[0], 1))
    # print(X, Y)
    # linear_regressor.fit(X, Y)  # perform linear regression
    # Y_pred = linear_regressor.predict(X)  # make predictions
    # print(linear_regressor.intercept_, linear_regressor.coef_)
    par = np.polyfit(X, Y, 1, full=True)
    m = par[0][0]
    b = par[0][1]
    print(np.round(m, 2), np.round(b, 2))
    fig, ax = plt.subplots()
    ax.scatter(X, Y)
    ax.plot(X, m*X + b)
    plt.show()

    X = np.concatenate(
        np.array([[k] * len(v) for (k, v) in allefficiency.items()]))
    Y = np.concatenate(np.array([v for _, v in allefficiency.items()]))
    par = np.polyfit(X, Y, 1, full=True)
    m = par[0][0]
    b = par[0][1]
    print('Efficiency', np.round(m, 2), np.round(b, 2))
    fig, ax = plt.subplots()
    ax.scatter(X, Y)
    ax.plot(X, m*X + b)
    plt.show()
    np.save(
        '/tmp/foraging_power_efficiency_ablation.npy',
        np.array([allpower, allefficiency], dtype=object))


def distortion_efficiency_power_foraging():
    ips = [0.8, 0.85, 0.9, 0.99]
    allpower = {}
    allefficiency = {}
    for ip in ips:
        data = np.squeeze(read_data_n_agent_perturbations_all(
            n=100, iter=12000, threshold=7, time=10000, iprob=ip,
            addobject=None, no_objects=1,
            radius=5, idx=[2]))

        powerdata100 = data[:, -1]
        allpower[ip] = powerdata100

        effdata = [np.squeeze(
            np.argwhere(data[i, :] >= 80)) for i in range(data.shape[0])]
        effdata = [np.array(
            [effdata[i][0] if effdata[i].shape[
                0] > 1 else 12000 for i in range(len(effdata))])]
        effdata = [(
            ((12000-effdata[i])/12000)*100) for i in range(len(effdata))]
        allefficiency[ip] = np.round(effdata[0], 2)

    X = np.concatenate(
        np.array([[i] * len(v[1]) for (i, v) in enumerate(allpower.items())]))
    Y = np.concatenate(np.array([v for _, v in allpower.items()]))

    par = np.polyfit(X, Y, 1, full=True)
    m = par[0][0]
    b = par[0][1]
    print(np.round(m, 2), np.round(b, 2))
    fig, ax = plt.subplots()
    ax.scatter(X, Y)
    ax.plot(X, m*X + b)
    plt.show()

    X = np.concatenate(
        np.array([[i] * len(
            v[1]) for (i, v) in enumerate(allefficiency.items())]))
    Y = np.concatenate(np.array([v for _, v in allefficiency.items()]))
    par = np.polyfit(X, Y, 1, full=True)
    m = par[0][0]
    b = par[0][1]
    print('Efficiency', np.round(m, 2), np.round(b, 2))
    fig, ax = plt.subplots()
    ax.scatter(X, Y)
    ax.plot(X, m*X + b)
    plt.show()
    np.save(
        '/tmp/foraging_power_efficiency_distortion.npy',
        np.array([allpower, allefficiency], dtype=object))


def shift_efficiency_power_foraging():
    timings = range(1000, 11001, 1000)
    allpower = {}
    allefficiency = {}
    for t in timings:
        data = np.squeeze(read_data_n_agent_perturbations_all(
            n=100, iter=12000, threshold=7, time=t, iprob=0.85,
            addobject='Obstacles', no_objects=2,
            radius=10, idx=[2]))

        powerdata100 = data[:, -1]
        allpower[t] = powerdata100

        effdata = [np.squeeze(
            np.argwhere(data[i, :] >= 80)) for i in range(data.shape[0])]
        effdata = [np.array(
            [effdata[i][0] if effdata[i].shape[
                0] > 1 else 12000 for i in range(len(effdata))])]
        effdata = [(
            ((12000-effdata[i])/12000)*100) for i in range(len(effdata))]
        allefficiency[t] = np.round(effdata[0], 2)

    X = np.concatenate(
        np.array([[i] * len(v[1]) for (i, v) in enumerate(allpower.items())]))
    Y = np.concatenate(np.array([v for _, v in allpower.items()]))

    par = np.polyfit(X, Y, 1, full=True)
    m = par[0][0]
    b = par[0][1]
    print(np.round(m, 2), np.round(b, 2))
    fig, ax = plt.subplots()
    ax.scatter(X, Y)
    ax.plot(X, m*X + b)
    plt.show()

    X = np.concatenate(
        np.array([[i] * len(
            v[1]) for (i, v) in enumerate(allefficiency.items())]))
    Y = np.concatenate(np.array([v for _, v in allefficiency.items()]))
    par = np.polyfit(X, Y, 1, full=True)
    m = par[0][0]
    b = par[0][1]
    print('Efficiency', np.round(m, 2), np.round(b, 2))
    fig, ax = plt.subplots()
    ax.scatter(X, Y)
    ax.plot(X, m*X + b)
    plt.show()
    np.save(
        '/tmp/foraging_power_efficiency_shift.npy',
        np.array([allpower, allefficiency], dtype=object))

def main():
    # ablation_efficiency_power_foraging()
    # distortion_efficiency_power_foraging()
    shift_efficiency_power_foraging()


if __name__ == "__main__":
    main()
