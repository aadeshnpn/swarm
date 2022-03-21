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


def plot_power_efficiency_subplots():
    TINY_SIZE = 7
    SMALL_SIZE = 8
    MEDIUM_SIZE = 10
    BIGGER_SIZE = 12

    plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=TINY_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=TINY_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=TINY_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=TINY_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=TINY_SIZE)  # fontsize of the figure title

    # Foraging data
    ablation = np.load(
        '/tmp/foraging_power_efficiency_ablation.npy', allow_pickle=True)
    addition = None
    distortion = np.load(
        '/tmp/foraging_power_efficiency_distortion.npy', allow_pickle=True)
    shift = np.load(
        '/tmp/foraging_power_efficiency_shift.npy', allow_pickle=True)

    fig = plt.figure(figsize=(6, 4), dpi=300)
    ax1 = fig.add_subplot(2, 4, 1)
    ax2 = fig.add_subplot(2, 4, 5)
    ax3 = fig.add_subplot(2, 4, 2)
    ax4 = fig.add_subplot(2, 4, 6)
    ax5 = fig.add_subplot(2, 4, 3)
    ax6 = fig.add_subplot(2, 4, 7)
    ax7 = fig.add_subplot(2, 4, 4)
    ax8 = fig.add_subplot(2, 4, 8)

    colors = ['cyan', 'magenta']
    # Ablation
    X = np.concatenate(
        np.array(
            [[i+1] * len(v[1]) for (i, v) in enumerate(
                ablation[0].items())]))
    Y = np.concatenate(np.array([v for _, v in ablation[0].items()]))
    # print(X)
    par = np.polyfit(X, Y, 1, full=True)
    m = par[0][0]
    b = par[0][1]
    print('Ablation power:', np.round(m, 2), np.round(b, 2))
    ax1.scatter(X, Y, color=colors[0], alpha=0.5, s=0.8, marker="8")
    ax1.plot(X, m*X + b, color=colors[0], linewidth=0.8)
    ax1.set_xticks(range(1, 6, 1))
    ax1.set_xticklabels(range(1, 6, 1))
    ax1.set_yticks(range(0, 101, 20))
    ax1.set_yticklabels(range(0, 101, 20))
    # ax1.set_xlabel('No. of Obstacles', fontsize="small")
    ax1.set_ylabel('Power')
    ax1.set_title('Ablation')

    X = np.concatenate(
        np.array(
            [[i+1] * len(v[1]) for (i, v) in enumerate(
                ablation[1].items())]))
    Y = np.concatenate(np.array([v for _, v in ablation[1].items()]))

    par = np.polyfit(X, Y, 1, full=True)
    m = par[0][0]
    b = par[0][1]
    print('Ablation efficiency:', np.round(m, 2), np.round(b, 2))
    ax2.scatter(X, Y, color=colors[0], alpha=0.5, s=0.8, marker="8")
    ax2.plot(X, m*X + b, color=colors[0], linewidth=0.8)

    ax2.set_xticks(range(1, 6, 1))
    ax2.set_xticklabels(range(1, 6, 1))
    ax2.set_yticks(range(0, 101, 20))
    ax2.set_yticklabels(range(0, 101, 20))
    ax2.set_xlabel('No. of Obstacles')
    ax2.set_ylabel('Efficiency')

    # Skip addition for now.

    # Distortion
    X = np.concatenate(
        np.array(
            [[i] * len(v[1]) for (i, v) in enumerate(
                distortion[0].items())]))
    Y = np.concatenate(np.array([v for _, v in distortion[0].items()]))

    par = np.polyfit(X, Y, 1, full=True)
    m = par[0][0]
    b = par[0][1]
    print('Distortion power:', np.round(m, 2), np.round(b, 2))
    ax5.scatter(X, Y, color=colors[0], alpha=0.5, s=0.8, marker="8")
    ax5.plot(X, m*X + b, color=colors[0], linewidth=0.8)
    ax5.set_xticks(range(0, 4, 1))
    ax5.set_xticklabels([0.8, 0.85, 0.9, 0.99])
    ax5.set_yticks(range(0, 101, 20))
    ax5.set_yticklabels(range(0, 101, 20))
    # 5x1.set_xlabel('No. of Obstacles', fontsize="small")
    # ax5.set_ylabel('Power')
    ax5.set_title('Distortion')

    X = np.concatenate(
        np.array(
            [[i] * len(v[1]) for (i, v) in enumerate(
                distortion[1].items())]))
    Y = np.concatenate(np.array([v for _, v in distortion[1].items()]))

    par = np.polyfit(X, Y, 1, full=True)
    m = par[0][0]
    b = par[0][1]
    print('Distortion efficiency:', np.round(m, 2), np.round(b, 2))
    ax6.scatter(X, Y, color=colors[0], alpha=0.5, s=0.8, marker="8")
    ax6.plot(X, m*X + b, color=colors[0], linewidth=0.8)
    ax6.set_xticks(range(0, 4, 1))
    ax6.set_xticklabels([0.8, 0.85, 0.9, 0.99])
    ax6.set_yticks(range(0, 101, 20))
    ax6.set_yticklabels(range(0, 101, 20))
    ax6.set_xlabel('IP')
    # ax6.set_ylabel('Efficiency')

    # Shift
    X = np.concatenate(
        np.array(
            [[i] * len(v[1]) for (i, v) in enumerate(
                shift[0].items())]))
    Y = np.concatenate(np.array([v for _, v in shift[0].items()]))

    par = np.polyfit(X, Y, 1, full=True)
    m = par[0][0]
    b = par[0][1]
    print('Shift power:', np.round(m, 2), np.round(b, 2))
    ax7.scatter(X, Y, color=colors[0], alpha=0.5, s=0.8, marker="8")
    ax7.plot(X, m*X + b, color=colors[0], linewidth=0.8)
    ax7.set_xticks(range(0, 11, 2))
    ax7.set_xticklabels([i if i==0 else str(i)+'k' for i in range(0, 11, 2)])
    ax7.set_yticks(range(0, 101, 20))
    ax7.set_yticklabels(range(0, 101, 20))
    # 7x1.set_xlabel('No. of Obstacles', fontsize="small")
    # ax7.set_ylabel('Power')
    ax7.set_title('Shift')

    X = np.concatenate(
        np.array(
            [[i] * len(v[1]) for (i, v) in enumerate(
                shift[1].items())]))
    Y = np.concatenate(np.array([v for _, v in shift[1].items()]))

    par = np.polyfit(X, Y, 1, full=True)
    m = par[0][0]
    b = par[0][1]
    print('Shift efficiency:', np.round(m, 2), np.round(b, 2))
    ax8.scatter(X, Y, color=colors[0], alpha=0.5, s=0.8, marker="8")
    ax8.plot(X, m*X + b, color=colors[0], linewidth=0.8)
    ax8.set_xticks(range(0, 11, 2))
    ax8.set_xticklabels([i if i==0 else str(i)+'k' for i in range(0, 11, 2)])
    ax8.set_yticks(range(0, 101, 20))
    ax8.set_yticklabels(range(0, 101, 20))
    ax8.set_xlabel('Timings')
    # ax8.set_ylabel('Efficiency')

    plt.tight_layout()
    maindir = '/tmp/swarm/data/experiments'
    fname = 'efficiency_power_all'

    fig.savefig(
        maindir + '/' + fname + '.png')
    # pylint: disable = E1101

    plt.close(fig)
    # Addition


def main():
    # ablation_efficiency_power_foraging()
    # distortion_efficiency_power_foraging()
    # shift_efficiency_power_foraging()
    plot_power_efficiency_subplots()


if __name__ == "__main__":
    main()
