"""Script to compute resilience metric"""
"""for all four types of perturbations for both swarm tasks."""
from audioop import add
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
from numpy.polynomial import Chebyshev


def read_data_n_agent_perturbations_all(
        n=100, agent="ExecutingAgent", expp=2, site=30,
        trap=5, obs=5, no_trap=0, no_obs=0, width=100, height=100, no_site=1, grid=10,
        idx=[2], fname='CoevoSimulation'):
    # maindir = '/tmp/swarm/data/experiments/' + fname
    nadir = os.path.join(
                '/tmp', 'swarm', 'data', 'experiments',
                str(n), agent, str(expp), str(site), str(trap) + '_' + str(obs),
                str(no_trap) + '_' + str(no_obs), str(width) + '_' + str(height), str(no_site), str(grid)
                )

    print(nadir, fname)
    folders = pathlib.Path(nadir).glob("*"+fname)
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


def ablation_efficiency_power(fname="CoevoSimulation"):
    t = 7
    no_obstacles = [1, 2, 3, 4, 5]

    allpower = {}
    allefficiency = {}

    for o in no_obstacles:
        data = [read_data_n_agent_perturbations_all(expp=4, no_obs=o, no_site=1, fname=fname)]
        # print(o, len(data), data[8:])
        powerdata100 = np.squeeze(data)[:, -1]
        print(o, powerdata100)
        # powerdata100 = [np.median(
        # powerdata100[i]) for i in range(len(powerdata100))]
        # allpower[o] = powerdata100
        allpower[o] = powerdata100

        effdata100 = []
        # print(o, data[t].shape)
        effdata = np.squeeze(data)
        effdata = [np.squeeze(
            np.argwhere(
                effdata[i, :] >= 80)) for i in range(effdata.shape[0])]
        effdata = [np.array(
            [effdata[i][
                0] if effdata[i].shape[
                    0] > 1 else 12000 for i in range(len(effdata))])]
        print('min', np.min(effdata))
        # Perturbations: 5279, 2756
        tmax = 12000
        tmin = np.min(effdata)  # 2756
        effdata = [(
            (
                1 - ((effdata[i] - tmin)/(
                    tmax-tmin)))*100) for i in range(len(effdata))]
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
        '/tmp/' + fname + '_power_efficiency_ablation.npy',
        np.array([allpower, allefficiency], dtype=object))


def distortion_efficiency_power(fname="CoevoSimulation"):
    grid = [2, 5, 10]
    allpower = {}
    allefficiency = {}
    for g in grid:
        data = read_data_n_agent_perturbations_all(
            expp=6, no_obs=0, no_site=1, grid=g, fname=fname)
        # print(o, len(data), data[8:])
        data = np.squeeze(data)
        powerdata100 = np.squeeze(data)[:, -1]
        allpower[g] = powerdata100

        effdata = [np.squeeze(
            np.argwhere(data[i, :] >= 80)) for i in range(data.shape[0])]
        effdata = [np.array(
            [effdata[i][0] if effdata[i].shape[
                0] > 1 else 12000 for i in range(len(effdata))])]
        print('min', np.min(effdata))
        # Min: 5200
        tmax = 12000
        tmin = np.min(effdata)     # 5200
        effdata = [(
            (
                1 - (abs(effdata[i] - tmin)/(
                    tmax-tmin)))*100) for i in range(len(effdata))]
        # effdata = [(
        #     ((12000-effdata[i])/12000)*100) for i in range(len(effdata))]
        allefficiency[g] = np.round(effdata[0], 2)

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
        '/tmp/' + fname + '_power_efficiency_distortion.npy',
        np.array([allpower, allefficiency], dtype=object))


def shift_efficiency_power(fname="CoevoSimulation"):
    sites = [1, 2, 3, 4, 5]
    allpower = {}
    allefficiency = {}
    for s in sites:
        data = read_data_n_agent_perturbations_all(
            expp=5, no_obs=0, no_site=s, grid=10, fname=fname)
        # print(o, len(data), data[8:])
        data = np.squeeze(data)
        powerdata100 = np.squeeze(data)[:, -1]
        allpower[s] = powerdata100
        effdata = [np.squeeze(
            np.argwhere(data[i, :] >= 80)) for i in range(data.shape[0])]
        effdata = [np.array(
            [effdata[i][0] if effdata[i].shape[
                0] > 1 else 12000 for i in range(len(effdata))])]
        print('min', np.min(effdata))
        # Min: 6778
        tmax = 12000
        tmin = np.min(effdata)     # 5200
        effdata = [(
            (
                1 - (abs(effdata[i] - tmin)/(
                    tmax-tmin)))*100) for i in range(len(effdata))]
        # effdata = [(
        #     ((12000-effdata[i])/12000)*100) for i in range(len(effdata))]
        allefficiency[s] = np.round(effdata[0], 2)

    X = np.concatenate(
        np.array([[i] * len(v[1]) for (i, v) in enumerate(allpower.items())]))
    Y = np.concatenate(np.array([v for _, v in allpower.items()]))
    # print(X.shape, Y.shape)

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
        '/tmp/' + fname + '_power_efficiency_shift.npy',
        np.array([allpower, allefficiency], dtype=object))


def addition_efficiency_power(fname="CoevoSimulation"):
    allpower = {}
    allefficiency = {}
    hitrate = []
    size = [100, 200, 300, 400, 500]
    for s in size:
        data = read_data_n_agent_perturbations_all(
            expp=3, no_obs=0, no_site=1, width=s, height=s, fname=fname)
        data = np.squeeze(data)
        # print(data.shape)
        powerdata100 = np.squeeze(data)[:, -1]
        allpower[s] = powerdata100

        effdata = [np.squeeze(
            np.argwhere(data[i, :] >= 80)) for i in range(data.shape[0])]
        effdata = [np.array(
            [effdata[i][0] if effdata[i].shape[
                0] > 1 else 12000 for i in range(len(effdata))])]
        hitrate.append(np.sum(effdata[0]!=12000)/effdata[0].shape[0])
        # print('min', np.min(effdata))
        # 5310
        tmax = 12000
        tmin = np.min(effdata)     # 5200
        effdata = [(
            (
                1 - (abs(effdata[i] - tmin)/(
                    tmax-tmin)))*100) if tmin!=12000 else [0] for i in range(len(effdata))]
        # print(effdata)
        # effdata = [(
        #     ((12000-effdata[i])/12000)*100) for i in range(len(effdata))]
        allefficiency[s] = np.round(effdata[0], 2)
    # print(hitrate)
    # print('Success Rate', np.round(np.median(np.array(hitrate)), 2))
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
        '/tmp/' + fname + '_power_efficiency_addition.npy',
        np.array([allpower, allefficiency], dtype=object))


def subplot_perturbations(
        data, paxis, xtick, xlabels,
        color, xlabel, pname='Ablation', metric='Power',
        forge=True, j=0):
    if forge:
        X = np.concatenate(
            np.array(
                [[i+j] * len(v[1][:40]) for (i, v) in enumerate(
                    data.items())]))
    else:
        X = np.concatenate(
            np.array(
                [[i+j+0.2] * len(v[1][:40]) for (i, v) in enumerate(
                    data.items())]))
    Y = np.concatenate(np.array([v[:40] for _, v in data.items()]))

    par = np.polyfit(X, Y, 1, full=True)
    m = par[0][0]
    b = par[0][1]
    print(pname + ' ' + metric + ' :', np.round(m, 2), np.round(b, 2), Y.shape)
    paxis.scatter(X, Y, color=color, alpha=0.5, s=0.8, marker="x")
    paxis.plot(X, m*X + b, color=color, linewidth=0.8)
    # paxisx1.set_xlabel('No. of Obstacles', fontsize="small")
    if pname == 'Ablation':
        paxis.set_ylabel(metric)
    if metric == 'Power':
        paxis.set_title(pname)
    if metric == 'Efficiency':
        paxis.set_xlabel(xlabel)

    paxis.set_xticks(xtick)
    paxis.set_xticklabels(xlabels)
    paxis.set_yticks(list(range(0, 101, 20)))
    paxis.set_yticklabels(list(range(0, 101, 20)))


def plot_power_efficiency_subplots():
    TINNY_SIZE = 5
    TINY_SIZE = 7
    SMALL_SIZE = 9
    MEDIUM_SIZE = 10
    BIGGER_SIZE = 12

    plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=SMALL_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=SMALL_SIZE)  # fontsize of the figure title

    # Foraging data
    ablation = np.load(
        '/tmp/CoevoSimulation_power_efficiency_ablation.npy',
        allow_pickle=True)
    addition = np.load(
        '/tmp/CoevoSimulation_power_efficiency_addition.npy',
        allow_pickle=True)
    distortion = np.load(
        '/tmp/CoevoSimulation_power_efficiency_distortion.npy',
        allow_pickle=True)
    shift = np.load(
        '/tmp/CoevoSimulation_power_efficiency_shift.npy',
        allow_pickle=True)

    distortion_nest = np.load(
        '/tmp/NestSimulation_power_efficiency_distortion.npy',
        allow_pickle=True)

    ablation_nest = np.load(
        '/tmp/NestSimulation_power_efficiency_ablation.npy',
        allow_pickle=True)

    shift_nest = np.load(
        '/tmp/NestSimulation_power_efficiency_shift.npy',
        allow_pickle=True)

    addition_nest = np.load(
        '/tmp/NestSimulation_power_efficiency_addition.npy',
        allow_pickle=True)

    fig = plt.figure(figsize=(6, 4), dpi=1600)
    ax1 = fig.add_subplot(2, 4, 1)
    ax2 = fig.add_subplot(2, 4, 5)
    ax3 = fig.add_subplot(2, 4, 2)
    ax4 = fig.add_subplot(2, 4, 6)
    ax5 = fig.add_subplot(2, 4, 3)
    ax6 = fig.add_subplot(2, 4, 7)
    ax7 = fig.add_subplot(2, 4, 4)
    ax8 = fig.add_subplot(2, 4, 8)

    colors = ['hotpink', 'mediumblue']

    # Shift
    xticks = [1, 2, 3, 4, 5]
    # Fixing the ytick issue with shit data
    ax8.scatter([1], [96], color="white", alpha=0.1, s=0.8, marker="x")

    subplot_perturbations(
        shift[0], ax7, range(1, 6), xticks,
        color=colors[0], xlabel='', pname='Shift II', metric='Power', j=1)
    subplot_perturbations(
        shift[1], ax8, range(1, 6), xticks,
        color=colors[0], xlabel='No. of Site',
        pname='Shift II', metric='Efficiency', j=1)

    subplot_perturbations(
        shift_nest[0], ax7, range(1, 6), xticks,
        color=colors[1], xlabel='', pname='Shift II',
        metric='Power', forge=False, j=1)
    subplot_perturbations(
        shift_nest[1], ax8, range(1, 6), xticks,
        color=colors[1], xlabel='No. of Site',
        pname='Shift II', metric='Efficiency', forge=False, j=1)

    # # Foraging Ablation
    subplot_perturbations(
        ablation[0], ax1, range(1, 6, 1), range(1, 6, 1),
        color=colors[0], xlabel='', pname='Ablation', metric='Power', j=1)
    subplot_perturbations(
        ablation[1], ax2, range(1, 6, 1), range(1, 6, 1),
        color=colors[0], xlabel='No. of Obstacles',
        pname='Ablation', metric='Efficiency', j=1)
    # Nest Ablation
    subplot_perturbations(
        ablation_nest[0], ax1, range(1, 6, 1), range(1, 6, 1),
        color=colors[1], xlabel='', pname='Ablation',
        metric='Power', forge=False, j=1)
    subplot_perturbations(
        ablation_nest[1], ax2, range(1, 6, 1), range(1, 6, 1),
        color=colors[1], xlabel='No. of Obstacles',
        pname='Ablation', metric='Efficiency', forge=False, j=1)

    # # Addition.
    # xticks = [i if i==0 else str(i)+'k' for i in range(1, 6, 1)]
    # addition[0] = {k:v for k,v in addition[0].items() if k in range(2000, 11000, 2000)}
    # addition[1] = {k:v for k,v in addition[1].items() if k in range(2000, 11000, 2000)}

    subplot_perturbations(
        addition[0], ax3, range(1, 6, 1), xticks,
        color=colors[0], xlabel='', pname='Addition II', metric='Power', j=1)
    subplot_perturbations(
        addition[1], ax4, range(1, 6, 1), xticks,
        color=colors[0], xlabel='Environment Size',
        pname='Addition II', metric='Efficiency', j=1)
    # Nest
    # addition_nest[0] = {k:v for k,v in addition_nest[0].items() if k in range(2000, 11000, 2000)}
    # addition_nest[1] = {k:v for k,v in addition_nest[1].items() if k in range(2000, 11000, 2000)}
    subplot_perturbations(
        addition_nest[0], ax3, range(1, 6, 1), xticks,
        color=colors[1], xlabel='', pname='Addition II',
        metric='Power', forge=False, j=1)
    subplot_perturbations(
        addition_nest[1], ax4, range(1, 6, 1), xticks,
        color=colors[1], xlabel='Environment Size',
        pname='Addition II', metric='Efficiency', forge=False, j=1)

    # # Distortion
    # # Foraging
    xticks = [2, 5, 10]
    subplot_perturbations(
        distortion[0], ax5, range(0, 3, 1), xticks,
        color=colors[0], xlabel='', pname='Distortion II', metric='Power')
    subplot_perturbations(
        distortion[1], ax6, range(0, 3, 1), xticks,
        color=colors[0], xlabel='Grid Size',
        pname='Distortion II', metric='Efficiency')
    # Nest
    subplot_perturbations(
        distortion_nest[0], ax5, range(0, 3, 1), xticks,
        color=colors[1], xlabel='',
        pname='Distortion II', metric='Power', forge=False)
    subplot_perturbations(
        distortion_nest[1], ax6, range(0, 3, 1), xticks,
        color=colors[1], xlabel='Grid Size',
        pname='Distortion II', metric='Efficiency', forge=False)

    f = lambda m,c: plt.plot([],[],marker=m, color=c, ls="none")[0]

    handles1 = [f('_', colors[i]) for i in range(2)]

    legend1 = ax1.legend(
        handles1, ['Foraging', 'Nest Maintenance'],
        # bbox_to_anchor=(0.05, 0.05, 0.05, 0.05),
        fontsize=TINNY_SIZE, loc="lower right", title='', markerscale=1)

    plt.setp(legend1.get_title(), fontsize='small')
    ax1.add_artist(legend1)

    plt.tight_layout()
    maindir = '/tmp/swarm/data/experiments'
    fname = 'efficiency_power_all_fixed_resilience'

    fig.savefig(
        maindir + '/' + fname + '.png')
    fig.savefig(
        maindir + '/' + fname + '.eps')
    # pylint: disable = E1101

    plt.close(fig)
    # Addition


def ip_ltr_gsr_relationship(fname="EvoCoevolutionPPA"):
    ips = [0.5, 0.7, 0.8, 0.85, 0.9, 0.99]
    allipsltr = {}
    allipgsr = {}
    for ip in ips:
        data = np.squeeze(read_data_n_agent_perturbations_all(
            n=100, iter=12000, threshold=7, time=10000, iprob=ip,
            addobject=None, no_objects=1,
            radius=5, idx=[4, 6], fname=fname))
        allipsltr[ip] = np.round(np.mean(data[0], axis=0), 0)[1:]
        allipgsr[ip] = np.round(np.mean(data[1], axis=0), 0)[1:]
        # print(ip, allipsltr[ip][0], allipsltr[ip][-1])

    X = np.concatenate(np.array([[i]*12001 for i in ips]))
    Y = np.concatenate(np.array([v for _, v in allipsltr.items()]))
    par = np.polyfit(X, Y, 1, full=True)
    m = par[0][0]
    b = par[0][1]
    print('IP VS LTR, slope, intercept', np.round(m, 2), np.round(b, 2))
    fig = plt.figure(figsize=(6, 4), dpi=300)
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.scatter(X, Y)
    ax1.plot(X, m*X + b)
    ax1.set_xticks(ips)
    ax1.set_xticklabels(ips)
    # plt.show()

    X = np.concatenate(np.array([[i]*12001 for i in ips]))
    Y = np.concatenate(np.array([v for _, v in allipgsr.items()]))
    par = np.polyfit(X, Y, 1, full=True)
    m = par[0][0]
    b = par[0][1]
    print('IP vs GSR, slope, intercept', np.round(m, 2), np.round(b, 2))
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.scatter(X, Y)
    ax2.plot(X, m*X + b)
    ax2.set_xticks(ips)
    ax2.set_xticklabels(ips)
    plt.show()
    plt.tight_layout()
    maindir = '/tmp/swarm/data/experiments'
    fname = 'ip_ltr_gsr'

    fig.savefig(
        maindir + '/' + fname + '.png')
    np.save(
        '/tmp/' + fname + '_ip_ltr_gsr.npy',
        np.array([X, allipsltr, allipgsr], dtype=object))


def st_ltr_gsr_relationship(fname="EvoCoevolutionPPA"):
    sts = [5, 7, 10, 15]
    allipsltr = {}
    allipgsr = {}
    for st in sts:
        data = np.squeeze(read_data_n_agent_perturbations_all(
            n=100, iter=12000, threshold=st, time=10000, iprob=0.85,
            addobject=None, no_objects=1,
            radius=5, idx=[4, 6], fname=fname))
        allipsltr[st] = np.round(np.mean(data[0], axis=0), 0)[1:]
        allipgsr[st] = np.round(np.mean(data[1], axis=0), 0)[1:]
        # print(ip, allipsltr[ip][0], allipsltr[ip][-1])

    X = np.concatenate(np.array([[i]*12001 for i in sts]))
    Y = np.concatenate(np.array([v for _, v in allipsltr.items()]))
    par = np.polyfit(X, Y, 1, full=True)
    m = par[0][0]
    b = par[0][1]
    print('ST VS LTR, slope, intercept', np.round(m, 2), np.round(b, 2))
    fig = plt.figure(figsize=(6, 4), dpi=300)
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.scatter(X, Y)
    ax1.plot(X, m*X + b)
    ax1.set_xticks(sts)
    ax1.set_xticklabels(sts)
    # plt.show()

    X = np.concatenate(np.array([[i]*12001 for i in sts]))
    Y = np.concatenate(np.array([v for _, v in allipgsr.items()]))
    par = np.polyfit(X, Y, 1, full=True)
    m = par[0][0]
    b = par[0][1]
    print('ST vs GSR, slope, intercept', np.round(m, 2), np.round(b, 2))
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.scatter(X, Y)
    ax2.plot(X, m*X + b)
    ax2.set_xticks(sts)
    ax2.set_xticklabels(sts)
    plt.show()
    plt.tight_layout()
    maindir = '/tmp/swarm/data/experiments'
    fname = 'st_ltr_gsr'

    fig.savefig(
        maindir + '/' + fname + '.png')
    np.save(
        '/tmp/' + fname + '_st_ltr_gsr.npy',
        np.array([X, allipsltr, allipgsr], dtype=object))


def power_efficiency_curve_lt_st_ltr_gs():
    pass


def main():
    ablation_efficiency_power()
    addition_efficiency_power()
    distortion_efficiency_power()
    shift_efficiency_power()
    # data = read_data_n_agent_perturbations_all(expp=4, no_obs=1, no_site=1)
    # print(data, data.shape)
    # distortion_efficiency_power()
    # addition_efficiency_power(fname="EvoCoevolutionPPA")
    # shift_efficiency_power()

    ablation_efficiency_power(fname="NestSimulation")
    distortion_efficiency_power(fname="NestSimulation")
    addition_efficiency_power(fname="NestSimulation")
    shift_efficiency_power(fname="NestSimulation")

    plot_power_efficiency_subplots()

    # addition_efficiency_power(fname="EvoCoevolutionPPAAd")
    # ip_ltr_gsr_relationship()
    # st_ltr_gsr_relationship()
    # Baseline : 5200


if __name__ == "__main__":
    main()
