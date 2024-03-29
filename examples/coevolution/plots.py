from cProfile import label
from email import header
from email.mime import base
import os
from pdb import post_mortem
import numpy as np
import scipy.stats as stats
import matplotlib

# If there is $DISPLAY, display the plot
if os.name == 'posix' and "DISPLAY" not in os.environ:
    matplotlib.use('Agg')

import matplotlib.pyplot as plt     # noqa: E402
import glob
import pathlib


def read_npy(fname='all.npy'):
    # /tmp/olddata/
    data = np.load('/tmp/olddata/' + fname)
    return data[:, -1]


def read_data_fitness(n=100, maindir='/tmp/div/diversity_withdecay'):
    # maindir = '/tmp/results_site_distance/experiments/'
    # maindir = '/home/aadeshnpn/Desktop/evolved_ppa/experiments/'
    # nadir = os.path.join(maindir, str(n), agent, str(site))
    folders = pathlib.Path(maindir).glob("*EvoSForgeNew*")
    flist = []
    dataf = []
    datad = []
    print(maindir)
    for f in folders:
        flist = [p for p in pathlib.Path(f).iterdir() if p.is_file() and p.match('simulation.csv')]
        try:
            _, _, f, d = np.genfromtxt(flist[0], autostrip=True, unpack=True, delimiter='|')
            dataf.append(f)
            datad.append(d)
            # print(flist[0], f.shape)
        except IndexError:
            pass
    # print(data)
    dataf = np.array(dataf)
    datad = np.array(datad)
    # print(dataf.shape, datad.shape, n)
    return dataf


# def read_data_n_agent(n=100, filename='/tmp/old.txt'):
#     maindir = '/tmp/swarm/data/experiments/'
#     files = np.genfromtxt(filename, unpack=True, autostrip=True, dtype=np.str)
#     data = []
#     for f in files:
#         # print(f)
#         _, _, d = np.genfromtxt(str(f), autostrip=True, unpack=True, delimiter='|')
#         data.append(d)
#     data = np.array(data)
#     # print(data.shape)
#     return data


def read_data_n_agent(n=100, iter=12000):
    maindir = '/tmp/swarm/data/experiments/EvoCoevolutionPPA/'
    nadir = os.path.join(maindir, str(n), str(iter))
    folders = pathlib.Path(nadir).glob("*EvoCoevolutionPPA")
    flist = []
    fdata = []
    mdata = []
    for f in folders:
        flist = [p for p in pathlib.Path(f).iterdir() if p.is_file() and p.match('simulation.csv')]
        _, _, f, m = np.genfromtxt(flist[0], autostrip=True, unpack=True, delimiter='|')
        fdata.append(f)
        mdata.append(m)
    fdata = np.array(fdata)
    mdata = np.array(mdata)
    print(fdata.shape, mdata.shape)
    return fdata, mdata


def read_data_n_agent_lt(n=100, iter=12000, lt='lt'):
    maindir = '/tmp/swarm/data/experiments/'
    nadir = os.path.join(maindir, str(n), str(iter), lt)
    # print(nadir)
    folders = pathlib.Path(nadir).glob("*EvoSForgeNewPPA1")
    flist = []
    fdata = []
    mdata = []
    for f in folders:
        flist = [p for p in pathlib.Path(f).iterdir() if p.is_file() and p.match('simulation.csv')]
        _, _, f, _ = np.genfromtxt(flist[0], autostrip=True, unpack=True, delimiter='|')
        fdata.append(f)
    fdata = np.array(fdata)
    return fdata


def read_data_n_agent_threshold(n=100, iter=12000, threshold=10):
    maindir = '/tmp/swarm/data/experiments/EvoCoevolutionPPA/'
    nadir = os.path.join(maindir, str(n), str(iter), str(threshold))
    print(nadir)
    folders = pathlib.Path(nadir).glob("*EvoCoevolutionPPA")
    flist = []
    fdata = []
    mdata = []
    for f in folders:
        flist = [p for p in pathlib.Path(f).iterdir() if p.is_file() and p.match('simulation.csv')]
        _, _, f, _ = np.genfromtxt(flist[0], autostrip=True, unpack=True, delimiter='|')
        # print(f.shape, flist[0])
        fdata.append(f)
    fdata = np.array(fdata)
    # print(fdata.shape)
    return fdata


def read_data_n_agent_threshold_new(n=100, iter=12000, threshold=10, gstep=200, expp=2):
    maindir = '/tmp/swarm/data/experiments/EvoCoevolutionPPA/'
    nadir = os.path.join(
        maindir, str(n), str(iter), str(threshold), str(gstep),
        str(expp), )
    print(nadir)
    folders = pathlib.Path(nadir).glob("*EvoCoevolutionPPA")
    flist = []
    fdata = []
    mdata = []
    for f in folders:
        flist = [p for p in pathlib.Path(f).iterdir() if p.is_file() and p.match('simulation.csv')]
        _, _, f, _ = np.genfromtxt(flist[0], autostrip=True, unpack=True, delimiter='|')
        # print(f.shape, flist[0])
        fdata.append(f)
    fdata = np.array(fdata)
    # print(fdata.shape)
    return fdata


def read_data_n_agent_perturbations(
        n=100, iter=12000, threshold=10, gstep=200, expp=2,
        addobject=None, removeobject=None, no_objects=1, radius=5,
        time=13000, iprob=0.85, idx=2):
    maindir = '/tmp/swarm/data/experiments/EvoCoevolutionPPA/'
    nadir = os.path.join(
                '/tmp', 'swarm', 'data', 'experiments', 'EvoCoevolutionPPA',
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
    for f in folders:
        flist = [p for p in pathlib.Path(f).iterdir() if p.is_file() and p.match('simulation.csv')]
        if len(flist) >=1:
            data = np.genfromtxt(flist[0], autostrip=True, unpack=True, delimiter='|')
            # print(f.shape, flist[0])
            fdata.append(data[idx])
    fdata = np.array(fdata)
    # print(fdata.shape)
    return fdata


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

    for f in folders:
        flist = [p for p in pathlib.Path(f).iterdir() if p.is_file() and p.match('simulation.csv')]
        if len(flist) > 0:
            data = np.genfromtxt(flist[0], autostrip=True, unpack=True, delimiter='|')
            # print(data.shape, flist[0])
            for i in range(len(idx)):
                fdata[i].append(data[idx[i]])
        # print(fdata)
    fdata = np.array(fdata)
    # print(fdata.shape)
    return fdata


def read_data_n_agent_perturbations_all_shift(
        n=100, iter=12000, threshold=10, gstep=200, expp=2,
        addobject=None, removeobject=None, no_objects=1, radius=5,
        time=13000, iprob=0.85, idx=[2], fname='EvoCoevolutionPPA',
        ltstopint=None):
    # maindir = '/tmp/swarm/data/experiments/' + fname
    if ltstopint is None:
        nadir = os.path.join(
                    '/tmp', 'swarm', 'data', 'experiments', fname,
                    str(n), str(iter), str(threshold), str(gstep), str(expp),
                    str(addobject), str(removeobject),
                    str(no_objects), str(radius),
                    str(time), str(iprob)
                    )
    else:
        nadir = os.path.join(
                    '/tmp', 'swarm', 'data', 'experiments', fname,
                    str(n), str(iter), str(threshold), str(gstep), str(expp),
                    str(addobject), str(removeobject),
                    str(no_objects), str(radius),
                    str(time), str(iprob), str(ltstopint)
                    )
    print(nadir)
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


def plot_none_obstacles_trap():
    fig = plt.figure(figsize=(8,6), dpi=200)
    none = read_data_n_agent_perturbations()
    obstacles = read_data_n_agent_perturbations(addobject='Obstacles', radius=10, time=2)
    # traps = read_data_n_agent_perturbations(addobject='Traps', radius=10, time=2)

    ax1 = fig.add_subplot(1, 1, 1)
    colordict = {
        0: 'gold',
        1: 'peru',
        2: 'orchid',
        3: 'olivedrab',
        4: 'linen',
        5: 'indianred',
        6: 'tomato'}
    colorshade = [
        'springgreen', 'lightcoral',
        'khaki', 'lightsalmon', 'deepskyblue']

    # labels = [str(a) for a in agent_sizes]
    medianprops = dict(linewidth=2.5, color='firebrick')
    meanprops = dict(linewidth=2.5, color='#ff7f0e')
    none_mean = np.median(none, axis=0)
    none_q1 = np.quantile(none, q =0.25, axis=0)
    none_q3 = np.quantile(none, q =0.75, axis=0)
    obstacles_mean = np.median(obstacles, axis=0)
    obs_q1 = np.quantile(obstacles, q =0.25, axis=0)
    obs_q3 = np.quantile(obstacles, q =0.75, axis=0)
    print(none_q3.shape)
    # traps_mean = np.mean(traps, axis=1)
    xvalues = range(12002)
    ax1.plot(xvalues, none_mean, label="None", color='blue')
    ax1.fill_between(
                xvalues, none_q1, none_q3, color='DodgerBlue', alpha=0.3)
    ax1.plot(xvalues, obstacles_mean, label="Obstacles", color='red')
    ax1.fill_between(
                xvalues, obs_q1, obs_q3, color='tomato', alpha=0.3)
    # ax1.plot(range(12001), traps_mean, labels="Traps")
    ax1.legend(fontsize="small", loc="upper right", title='Objects')

    # ax1.set_xticklabels(['Foraging', 'Maintenance'])
    ax1.set_xlabel('Evolution Steps')
    ax1.set_ylabel('Foraing (%)')
    ax1.set_yticks(range(0, 105, 20))

    plt.tight_layout()
    maindir = '/tmp/swarm/data/experiments/'
    # fname = 'agentsitecomp' + agent
    nadir = os.path.join(maindir, str(50))

    fig.savefig(
        nadir + 'coevolutionobjects' + '.png')
    plt.close(fig)


def plot_obstacles_time():
    fig = plt.figure(figsize=(8,6), dpi=200)
    time = [1000, 2000, 3000, 4000, 5000, 6000, 7000]
    obstaclestime = [read_data_n_agent_perturbations(addobject='Obstacles', no_objects=5, radius=10, time=t) for t in time]

    ax1 = fig.add_subplot(1, 1, 1)

    median = [ np.median(obstaclestime[i], axis=0) for i in range(len(time))]
    q1 = [ np.quantile(obstaclestime[i], q=0.25, axis=0) for i in range(len(time))]
    q3 = [ np.quantile(obstaclestime[i], q=0.75, axis=0) for i in range(len(time))]


    # traps_mean = np.mean(traps, axis=1)
    xvalues = range(12002)
    for i in range(len(time)):
        ax1.plot(xvalues, median[i], label=str(time[i]))
        # ax1.fill_between(
        #            xvalues, q1[i], q3[i],  alpha=0.3)

    # ax1.plot(range(12001), traps_mean, labels="Traps")
    ax1.legend(fontsize="small", loc="upper left", title='Perturbation Timestep')

    # ax1.set_xticklabels(['Foraging', 'Maintenance'])
    ax1.set_xlabel('Evolution Steps')
    ax1.set_ylabel('Foraing (%)')
    ax1.set_yticks(range(0, 105, 20))

    plt.tight_layout()
    maindir = '/tmp/swarm/data/experiments/'
    # fname = 'agentsitecomp' + agent
    nadir = os.path.join(maindir, str(50))

    fig.savefig(
        nadir + 'coevolutionobstaclestime' + '.png')
    plt.close(fig)


def plot_lt_rate():
    fig = plt.figure(figsize=(8,6), dpi=200)
    time = [1000, 2000, 3000, 4000, 5000, 6000, 7000]
    # obstaclestime = [read_data_n_agent_perturbations(addobject='Obstacles', no_objects=5, radius=10, time=t) for t in time]
    _,_,lowf,_,low = np.genfromtxt('/tmp/low1.csv', autostrip=True, unpack=True, delimiter='|')
    _,_,highf,_,high = np.genfromtxt('/tmp/high1.csv', autostrip=True, unpack=True, delimiter='|')
    ax1 = fig.add_subplot(1, 1, 1)
    xvalues = range(12002)
    ax1.scatter(xvalues, low,label='Worst Collective', alpha=0.2, s=5)
    ax1.scatter(xvalues, high,label='Best Collective', alpha=0.2, s=5)
    ax1.legend(fontsize="small", loc="upper left", title='LT Rate')
    ax1.set_xlabel('Evolution Steps')
    ax1.set_ylabel('LT Rate')
    ax1.set_yticks(range(0, 105, 20))

    ax2 = ax1.twinx()
    ax2.plot(xvalues, lowf, label='Low Foraging (%)', linewidth=2)
    ax2.plot(xvalues, highf, label='High Foraging (%)', linewidth=2)
    # traps_mean = np.mean(traps, axis=1)
    ax2.legend(fontsize="small", loc="upper right", title='Foraging (%)')

    # ax1.set_xticklabels(['Foraging', 'Maintenance'])
    ax2.set_xlabel('Evolution Steps')
    ax2.set_ylabel('Foraging %')
    ax2.set_yticks(range(0, 105, 20))

    plt.tight_layout()
    maindir = '/tmp/'
    # fname = 'agentsitecomp' + agent
    nadir = os.path.join(maindir, str(50))

    fig.savefig(
        nadir + 'coevolutionltrate' + '.png')
    plt.close(fig)


def read_data_n_agent_site(n=100, agent='ExecutingAgent', site='20'):
    maindir = '/tmp/results_site_distance/experiments/'
    # maindir = '/home/aadeshnpn/Desktop/evolved_ppa/experiments/'
    nadir = os.path.join(maindir, str(n), agent, str(site))
    folders = pathlib.Path(nadir).glob("*ForagingSim*")
    flist = []
    dataf = []
    datad = []
    for f in folders:
        flist = [p for p in pathlib.Path(f).iterdir() if p.is_file()]
        try:
            # print(flist)
            _, _, f, d = np.genfromtxt(flist[0], autostrip=True, unpack=True, delimiter='|')
            dataf.append(f)
            datad.append(d)
        except IndexError:
            pass
    # print(data)
    dataf = np.array(dataf)
    datad = np.array(datad)
    print(dataf.shape, datad.shape, n)
    return dataf, datad


def read_data_time(n=100):
    maindir = '/home/aadeshnpn/Desktop/experiments/'
    # nadir = os.path.join(maindir, str(n), str(iter))
    # folders = pathlib.Path(nadir).glob("*EvoSForgeNew*")
    data = np.genfromtxt(maindir + str(n) +'.txt', autostrip=True, unpack=True)
    data = np.array(data)
    print(data.shape)
    return data


def read_data_n_agent_6000(n=100, iter=6000):
    maindir = '/home/aadeshnpn/Desktop/experiments/'
    nadir = os.path.join(maindir, str(n), str(iter))
    folders = pathlib.Path(nadir).glob("*EvoSForgeNew*")
    flist = []
    data = []
    for f in folders:
        flist = [p for p in pathlib.Path(f).iterdir() if p.is_file() and p.match('simulation.csv')]
        # print(flist)
        _, _, d, _ = np.genfromtxt(flist[0], autostrip=True, unpack=True, delimiter='|')
        # print(d.shape)
        data.append(d)
    data = np.array(data)
    print(data.shape)
    return data


def plot_evolution_algo_performance():
    # plt.style.use('fivethirtyeight')
    agent_sizes = [50, 100, 150, 200]
    # dataf = [read_data_n_agent_6000(n=a)[:,-1] for a in agent_sizes]
    # datat = [read_data_time(n=a) for a in agent_sizes]


    fig = plt.figure(figsize=(8,6), dpi=200)
    ax1 = fig.add_subplot(1, 1, 1)
    colordict = {
        0: 'gold',
        1: 'linen',
        2: 'orchid',
        3: 'peru',
        4: 'olivedrab',
        5: 'indianred',
        6: 'tomato'}
    colorshade = [
        'springgreen', 'lightcoral',
        'khaki', 'lightsalmon', 'deepskyblue']

    labels = [str(a) for a in agent_sizes]
    medianprops = dict(linewidth=2.5, color='firebrick')
    meanprops = dict(linewidth=2.5, color='#ff7f0e')
    # data = [data[:, i] for i in range(4)]
    positions = [
        [1], [4], [7], [10]
        ]
    # print(len(values_data), len(runtime_data))
    datas = [
        [dataf[0]],
        [dataf[1]],
        [dataf[2]],
        [dataf[3]],
    ]

    for j in range(len(positions)):
        bp1 = ax1.boxplot(
            datas[j], 0, 'gD', showmeans=True, meanline=True,
            patch_artist=True, medianprops=medianprops,
            meanprops=meanprops, positions=positions[j], widths=0.8)
        for patch, color in zip(bp1['boxes'], colordict.values()):
            patch.set_facecolor('gold')
        # plt.xlim(0, len(mean))
    ax2 = ax1.twinx()
    positions = [
        [2], [5], [8], [11]
        ]
    datas = [
        [datat[0]],
        [datat[1]],
        [datat[2]],
        [datat[3]],
    ]

    for j in range(len(positions)):
        bp2 = ax2.boxplot(
            datas[j], 0, 'gD', showmeans=True, meanline=True,
            patch_artist=True, medianprops=medianprops,
            meanprops=meanprops, positions=positions[j], widths=0.8)
        for patch, color in zip(bp2['boxes'], colordict.values()):
            patch.set_facecolor('deepskyblue')

    ax1.legend([bp1['boxes'][0], bp2['boxes'][0]], ['Foraing (%)', 'Runtime (secs)'], fontsize="small", loc="upper left", title='Performance Metric')
    # ax2.legend(zip(bp2['boxes']), ['Runtime (secs)'], fontsize="small", loc="lower right", title='Performance Measures')
    ax1.set_xticks(
        [1.5, 4.5, 7.5, 10.5
         ])
    ax1.set_xticklabels(labels)
    ax1.set_xlabel('Agent size', fontsize="large")
    ax1.set_ylabel('Foraing (%)', fontsize="large")
    ax1.set_yticks(range(0, 105, 20))

    ax2.set_ylabel('Runtime (Secs)', fontsize="large")
    ax2.set_yticks(range(0, 10000, 2000))

    plt.tight_layout()
    maindir = '/tmp/swarm/data/experiments/'
    # fname = 'agentsitecomp' + agent
    nadir = os.path.join(maindir, str(50))

    fig.savefig(
        nadir + 'evolutionperform2axes' + '.png')
    plt.close(fig)


def plot_evolution_algo_performance_boxplot():
    fig = plt.figure(figsize=(8,6), dpi=200)
    fdata, mdata = read_data_n_agent()
    # fdata = fdata[:, -1]
    # mdata = mdata[:, -1]
    fdata = np.max(fdata[:,1:], axis=1)
    mdata = np.max(mdata[:,1:], axis=1)
    print(fdata.shape, mdata.shape)
    print(fdata, mdata)
    ax1 = fig.add_subplot(1, 1, 1)
    colordict = {
        0: 'gold',
        1: 'peru',
        2: 'orchid',
        3: 'olivedrab',
        4: 'linen',
        5: 'indianred',
        6: 'tomato'}
    colorshade = [
        'springgreen', 'lightcoral',
        'khaki', 'lightsalmon', 'deepskyblue']

    # labels = [str(a) for a in agent_sizes]
    medianprops = dict(linewidth=2.5, color='firebrick')
    meanprops = dict(linewidth=2.5, color='#ff7f0e')

    bp1 = ax1.boxplot(
        [fdata, mdata], 0, 'gD', showmeans=True, meanline=True,
        patch_artist=True, medianprops=medianprops,
        meanprops=meanprops)
    for patch, color in zip(bp1['boxes'], colordict.values()):
        patch.set_facecolor(color)
        # plt.xlim(0, len(mean))
    ax1.legend(zip(bp1['boxes']), ['Foraing (%)', 'Maintenance (%)'], fontsize="small", loc="upper right", title='Performance Measures')
    # ax1.set_xticks(
    #     [1.5, 4.5, 7.5, 10.5
    #      ])
    ax1.set_xticklabels(['Foraging', 'Maintenance'])
    ax1.set_xlabel('Evolution Efficiency')
    ax1.set_ylabel('Foraing (%) / Maintenace (%)')
    ax1.set_yticks(range(0, 105, 20))

    plt.tight_layout()
    maindir = '/tmp/swarm/data/experiments/'
    # fname = 'agentsitecomp' + agent
    nadir = os.path.join(maindir, str(50))

    fig.savefig(
        nadir + 'coevolutionperform' + '.png')
    plt.close(fig)


def read_data_sample_ratio(ratio=0.1):
    # maindir = '/tmp/swarm/data/experiments/behavior_sampling'
    maindir = '/tmp/bsample/'
    ## Experiment ID for the plots/results in the paper
    # maindir = '/tmp/16244729911974EvoSForgeNewPPA1/'
    # maindir = '/tmp/experiments/100/12000/16243666378807EvoSForgeNewPPA1'
    # nadir = os.path.join(maindir, str(n), agent)
    # maindir = '/tmp/bsampling/'  # New sampling behaviors

    folders = pathlib.Path(maindir).glob("*_" + str(ratio) + "_ValidateSForgeNewPPA1")
    flist = []
    data = []
    for f in folders:
        # print(f)
        try:
            flist = [p for p in pathlib.Path(f).iterdir() if p.is_file() and p.match('simulation.csv')]
            _, _, d = np.genfromtxt(flist[0], autostrip=True, unpack=True, delimiter='|')
            # print(d.shape)
            data.append(d[-1])
        except:
            pass
    data = np.array(data)
    # print(ratio, data.shape)
    return data


def read_data_sample_ratio_ijcai(ratio=0.1):
    # maindir = '/tmp/swarm/data/experiments/behavior_sampling'
    maindir = '/tmp/ratioijcai/'
    ## Experiment ID for the plots/results in the paper
    # maindir = '/tmp/16244729911974EvoSForgeNewPPA1/'
    # maindir = '/tmp/experiments/100/12000/16243666378807EvoSForgeNewPPA1'
    # nadir = os.path.join(maindir, str(n), agent)
    # maindir = '/tmp/bsampling/'  # New sampling behaviors

    folders = pathlib.Path(maindir).glob("*[0-9]*-" + str(ratio))
    flist = []
    data = []
    for f in folders:
        # print(f)
        try:
            flist = [p for p in pathlib.Path(f).iterdir() if p.is_file() and p.match('simulation.csv')]
            _, _, d = np.genfromtxt(flist[0], autostrip=True, unpack=True, delimiter='|')
            # print(d.shape)
            data.append(d[-1])
        except:
            pass
    data = np.array(data)
    # print(ratio, data.shape)
    return data




def read_data_n(n=100, comm=True):
    maindir = '/tmp/swarm/data/experiments/'
    if comm:
        folders = pathlib.Path(maindir + 'comm').glob('*SForgeNewPPAComm1')
    else:
        folders = pathlib.Path(maindir + 'withoutcomm').glob('*ForagingSimulation')
    flist = []
    dataf = []
    datad = []
    for f in folders:
        try:
            flist = [p for p in pathlib.Path(f).iterdir() if p.is_file() and p.match('simulation.csv')]
            # print(flist)
            _, _, f, d = np.genfromtxt(flist[0], autostrip=True, unpack=True, delimiter='|')
            dataf.append(f)
            datad.append(d)
        except:
            pass
    dataf = np.array(dataf)
    datad = np.array(datad)
    return dataf, datad


def read_data_exp_3(width=100, height=100, trap=5, obs=5, exp_no=3, site=30, no_trap=1, no_obs=1, agent=100, grid=None, no_site=1):
    maindir = '/tmp/betrgeese0-4/'
    if grid is None:
        ndir = os.path.join(maindir, str(agent), 'ExecutingAgent', str(exp_no),
                str(site), str(trap)+'_'+str(obs), str(no_trap)+'_'+str(no_obs), str(width) +'_'+str(height), str(no_site))
    else:
        ndir = os.path.join(maindir, str(agent), 'ExecutingAgent', str(exp_no),
                str(site), str(trap)+'_'+str(obs), str(no_trap)+'_'+str(no_obs), str(width) +'_'+str(height), str(no_site), str(grid))
    print(ndir)
    folders = pathlib.Path(ndir).glob('*ForagingSimulation')
    flist = []
    dataf = []
    datad = []
    for f in folders:
        try:
            flist = [p for p in pathlib.Path(f).iterdir() if p.is_file() and p.match('simulation.csv')]
            # print(flist)
            _, _, f, d = np.genfromtxt(flist[0], autostrip=True, unpack=True, delimiter='|')
            dataf.append(f)
            datad.append(d)
        except:
            pass
    dataf = np.array(dataf)
    datad = np.array(datad)
    return dataf, datad


def read_data_exp_3_bt(width=100, height=100, trap=0, obs=0, exp_no=3, site=30, no_trap=0, no_obs=0, agent=100, grid=None, no_site=1):
    maindir = '/tmp/geesebt0-4/'
    if grid is None:
        ndir = os.path.join(maindir, str(agent), 'ExecutingAgent', str(exp_no),
                str(site), str(trap)+'_'+str(obs), str(no_trap)+'_'+str(no_obs), str(width) +'_'+str(height), str(no_site))
    else:
        ndir = os.path.join(maindir, str(agent), 'ExecutingAgent', str(exp_no),
                str(site), str(trap)+'_'+str(obs), str(no_trap)+'_'+str(no_obs), str(width) +'_'+str(height), str(no_site), str(grid))
    print(ndir)
    folders = pathlib.Path(ndir).glob('*ForagingSimulationOld')
    flist = []
    dataf = []
    # datad = []
    for f in folders:
        try:
            flist = [p for p in pathlib.Path(f).iterdir() if p.is_file() and p.match('simulation.csv')]
            # print(flist)
            _, _, forgingp = np.genfromtxt(flist[0], autostrip=True, unpack=True, delimiter='|')
            # print(f, forgingp.shape)
            dataf.append(forgingp)
            # datad.append(d)
        except:
            pass
    dataf = np.array(dataf)
    # datad = np.array(datad)
    return dataf

## Command to count the occurance of PPA sub-tree in each evolved behavior
# find $PWD -type f -name "*.json" | xargs cat | awk -F ',' '{for(i=1;i<=NF;i++){print $i;}}'
# | awk -F'<Act>' 'NF{print NF-1}' | awk '{a[$1]++}END{for(x in a)print a[x]"="x}'

def withWithoutLt():
    fig = plt.figure(figsize=(8,6), dpi=200)
    ltdata = read_data_n_agent_lt(n=50, iter=12000, lt='lt')
    noltdata = read_data_n_agent_lt(n=50, iter=12000, lt='nolt')
    # print(ltdata)
    ltdata = ltdata[:, -1]
    noltdata = noltdata[:, -1]
    # fdata = np.max(fdata[:,1:], axis=1)
    # mdata = np.max(mdata[:,1:], axis=1)
    # print(ltdata, noltdata)
    ax1 = fig.add_subplot(1, 1, 1)
    colordict = {
        5: 'gold',
        1: 'peru',
        2: 'orchid',
        3: 'olivedrab',
        4: 'linen',
        0: 'indianred',
        6: 'tomato'}
    colorshade = [
        'springgreen', 'lightcoral',
        'khaki', 'lightsalmon', 'deepskyblue']

    # labels = [str(a) for a in agent_sizes]
    medianprops = dict(linewidth=2.5, color='firebrick')
    meanprops = dict(linewidth=2.5, color='#ff7f0e')

    bp1 = ax1.boxplot(
        [noltdata, ltdata], 0, 'gD', showmeans=True, meanline=True,
        patch_artist=True, medianprops=medianprops,
        meanprops=meanprops)
    for patch, color in zip(bp1['boxes'], colordict.values()):
        patch.set_facecolor(color)
        # plt.xlim(0, len(mean))
    ax1.legend(zip(bp1['boxes']), ['Disabled', 'Enabled'], fontsize="small", loc="upper right", title='Lateral Transfer')
    # ax1.set_xticks(
    #     [1.5, 4.5, 7.5, 10.5
    #      ])
    ax1.set_xticklabels(['Disabled', 'Enabled'])
    ax1.set_xlabel('Lateral Transfer')
    ax1.set_ylabel('Foraing (%)')
    ax1.set_yticks(range(0, 105, 20))

    plt.tight_layout()
    maindir = '/tmp/swarm/data/experiments/'
    # fname = 'agentsitecomp' + agent
    nadir = os.path.join(maindir, str(50))

    fig.savefig(
        nadir + 'lateraltransferperform' + '.png')
    plt.close(fig)


def storage_threshold_new():
    thresholds = [5, 7, 11, 13, 17]
    data50 = [read_data_n_agent_threshold_new(n=50, iter=12000, threshold=t, gstep=200, expp=2)[:,-1] for t in thresholds]
    # data100 = [read_data_n_agent_threshold(n=100, iter=12000, threshold=t)[:,-1] for t in thresholds]

    fig = plt.figure(figsize=(8,6), dpi=200)
    ax1 = fig.add_subplot(1, 1, 1)
    colordict = {
        0: 'gold',
        1: 'linen',
        2: 'orchid',
        3: 'peru',
        4: 'olivedrab',
        5: 'indianred',
        6: 'tomato'}
    colorshade = [
        'springgreen', 'lightcoral',
        'khaki', 'lightsalmon', 'deepskyblue']

    # labels = [ "> n/"+str(a) for a in thresholds]
    medianprops = dict(linewidth=2.5, color='firebrick')
    meanprops = dict(linewidth=2.5, color='#ff7f0e')
    positions = [
        [1, 2], [4, 5], [7, 8], [10, 11], [13, 14]
        ]
    # datas = [
    #     [data50[0], data100[0]],
    #     [data50[1], data100[1]],
    #     [data50[2], data100[2]],
    #     [data50[3], data100[3]],
    #     [data50[4], data100[4]],
    #     # [np.zeros(np.shape(datas50[4])), data100[5]],
    # ]

    # for j in range(len(positions)):
    bp1 = ax1.boxplot(
        data50, 0, 'gD', showmeans=True, meanline=True,
        patch_artist=True, medianprops=medianprops,
        meanprops=meanprops, widths=0.8)
    for patch, color in zip(bp1['boxes'], colordict.values()):
        patch.set_facecolor(color)

    # ax1.legend(zip(bp1['boxes']), ['50', '100'], fontsize="small", loc="upper left", title='Agent Population (n)')
    ax1.legend(zip(bp1['boxes']), thresholds, fontsize="small", loc="upper right", title='Storage Threshold')
    # ax1.set_xticks([1.5, 4.5, 7.5, 10.5, 13.5])
    ax1.set_xticklabels(thresholds)
    ax1.set_yticks(range(0, 105, 20))
    ax1.set_xlabel('Storage Threshold', fontsize="large")
    ax1.set_ylabel('Foraging (%)', fontsize="large")
    plt.tight_layout()

    maindir = '/tmp/swarm/data/experiments'
    fname = 'thresholdboxplotnew'

    fig.savefig(
        maindir + '/' + fname + '.png')
    # pylint: disable = E1101

    plt.close(fig)


def exploration_parameter():
    exploration = [1, 2, 3, 4, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 47, 50, 60, 70, 80]
    data50 = [read_data_n_agent_threshold_new(n=50, iter=12000, threshold=5, gstep=200, expp=e)[:,-1] for e in exploration]
    # data100 = [read_data_n_agent_threshold(n=100, iter=12000, threshold=t)[:,-1] for t in thresholds]

    fig = plt.figure(figsize=(8,6), dpi=200)
    ax1 = fig.add_subplot(1, 1, 1)
    colordict = {
        0: 'gold',
        1: 'linen',
        2: 'orchid',
        3: 'peru',
        4: 'olivedrab',
        5: 'indianred',
        6: 'tomato'}
    colorshade = [
        'springgreen', 'lightcoral',
        'khaki', 'lightsalmon', 'deepskyblue']

    # labels = [ "> n/"+str(a) for a in thresholds]
    medianprops = dict(linewidth=2.5, color='firebrick')
    meanprops = dict(linewidth=2.5, color='#ff7f0e')
    positions = [
        [1, 2], [4, 5], [7, 8], [10, 11], [13, 14]
        ]
    # datas = [
    #     [data50[0], data100[0]],
    #     [data50[1], data100[1]],
    #     [data50[2], data100[2]],
    #     [data50[3], data100[3]],
    #     [data50[4], data100[4]],
    #     # [np.zeros(np.shape(datas50[4])), data100[5]],
    # ]

    # for j in range(len(positions)):
    bp1 = ax1.boxplot(
        data50, 0, 'gD', showmeans=True, meanline=True,
        patch_artist=True, medianprops=medianprops,
        meanprops=meanprops, widths=0.8)
    for patch, color in zip(bp1['boxes'], colordict.values()):
        patch.set_facecolor(color)

    # ax1.legend(zip(bp1['boxes']), ['50', '100'], fontsize="small", loc="upper left", title='Agent Population (n)')
    ax1.legend(zip(bp1['boxes']), exploration, fontsize="small", loc="upper right", title='Exploration')
    # ax1.set_xticks([1.5, 4.5, 7.5, 10.5, 13.5])
    ax1.set_xticklabels(exploration)
    ax1.set_yticks(range(0, 105, 20))
    ax1.set_xlabel('Exploration', fontsize="large")
    ax1.set_ylabel('Foraging (%)', fontsize="large")
    plt.tight_layout()

    maindir = '/tmp/swarm/data/experiments'
    fname = 'explorationboxplotnew'

    fig.savefig(
        maindir + '/' + fname + '.png')
    # pylint: disable = E1101

    plt.close(fig)


def generationstep_parameter():
    gsteps = [25, 50, 75, 100, 150, 200, 250, 300, 400, 500]
    data50 = [read_data_n_agent_threshold_new(n=50, iter=12000, threshold=5, gstep=g, expp=2)[:,-1] for g in gsteps]
    # data100 = [read_data_n_agent_threshold(n=100, iter=12000, threshold=t)[:,-1] for t in thresholds]

    fig = plt.figure(figsize=(8,6), dpi=200)
    ax1 = fig.add_subplot(1, 1, 1)
    colordict = {
        0: 'gold',
        1: 'linen',
        2: 'orchid',
        3: 'peru',
        4: 'olivedrab',
        5: 'indianred',
        6: 'tomato'}
    colorshade = [
        'springgreen', 'lightcoral',
        'khaki', 'lightsalmon', 'deepskyblue']

    # labels = [ "> n/"+str(a) for a in thresholds]
    medianprops = dict(linewidth=2.5, color='firebrick')
    meanprops = dict(linewidth=2.5, color='#ff7f0e')
    positions = [
        [1, 2], [4, 5], [7, 8], [10, 11], [13, 14]
        ]
    # datas = [
    #     [data50[0], data100[0]],
    #     [data50[1], data100[1]],
    #     [data50[2], data100[2]],
    #     [data50[3], data100[3]],
    #     [data50[4], data100[4]],
    #     # [np.zeros(np.shape(datas50[4])), data100[5]],
    # ]

    # for j in range(len(positions)):
    bp1 = ax1.boxplot(
        data50, 0, 'gD', showmeans=True, meanline=True,
        patch_artist=True, medianprops=medianprops,
        meanprops=meanprops, widths=0.8)
    for patch, color in zip(bp1['boxes'], colordict.values()):
        patch.set_facecolor(color)

    # ax1.legend(zip(bp1['boxes']), ['50', '100'], fontsize="small", loc="upper left", title='Agent Population (n)')
    ax1.legend(zip(bp1['boxes']), gsteps, fontsize="small", loc="upper right", title='GenerationSteps')
    # ax1.set_xticks([1.5, 4.5, 7.5, 10.5, 13.5])
    ax1.set_xticklabels(gsteps)
    ax1.set_yticks(range(0, 105, 20))
    ax1.set_xlabel('Generation Steps', fontsize="large")
    ax1.set_ylabel('Foraging (%)', fontsize="large")
    plt.tight_layout()

    maindir = '/tmp/swarm/data/experiments'
    fname = 'generationstepboxplotnew'

    fig.savefig(
        maindir + '/' + fname + '.png')
    # pylint: disable = E1101

    plt.close(fig)


def storage_threshold():
    thresholds = [5, 7, 11, 13, 17]
    data50 = [read_data_n_agent_threshold(n=50, iter=12000, threshold=t)[:,-1] for t in thresholds]
    # data100 = [read_data_n_agent_threshold(n=100, iter=12000, threshold=t)[:,-1] for t in thresholds]

    fig = plt.figure(figsize=(8,6), dpi=200)
    ax1 = fig.add_subplot(1, 1, 1)
    colordict = {
        0: 'gold',
        1: 'linen',
        2: 'orchid',
        3: 'peru',
        4: 'olivedrab',
        5: 'indianred',
        6: 'tomato'}
    colorshade = [
        'springgreen', 'lightcoral',
        'khaki', 'lightsalmon', 'deepskyblue']

    # labels = [ "> n/"+str(a) for a in thresholds]
    medianprops = dict(linewidth=2.5, color='firebrick')
    meanprops = dict(linewidth=2.5, color='#ff7f0e')
    positions = [
        [1, 2], [4, 5], [7, 8], [10, 11], [13, 14]
        ]
    # datas = [
    #     [data50[0], data100[0]],
    #     [data50[1], data100[1]],
    #     [data50[2], data100[2]],
    #     [data50[3], data100[3]],
    #     [data50[4], data100[4]],
    #     # [np.zeros(np.shape(datas50[4])), data100[5]],
    # ]

    # for j in range(len(positions)):
    bp1 = ax1.boxplot(
        data50, 0, 'gD', showmeans=True, meanline=True,
        patch_artist=True, medianprops=medianprops,
        meanprops=meanprops, widths=0.8)
    for patch, color in zip(bp1['boxes'], colordict.values()):
        patch.set_facecolor(color)

    # ax1.legend(zip(bp1['boxes']), ['50', '100'], fontsize="small", loc="upper left", title='Agent Population (n)')
    ax1.legend(zip(bp1['boxes']), thresholds, fontsize="small", loc="upper right", title='Storage Threshold')
    # ax1.set_xticks([1.5, 4.5, 7.5, 10.5, 13.5])
    ax1.set_xticklabels(thresholds)
    ax1.set_yticks(range(0, 105, 20))
    ax1.set_xlabel('Storage Threshold', fontsize="large")
    ax1.set_ylabel('Foraging (%)', fontsize="large")
    plt.tight_layout()

    maindir = '/tmp/swarm/data/experiments'
    fname = 'thresholdboxplot'

    fig.savefig(
        maindir + '/' + fname + '.png')
    # pylint: disable = E1101

    plt.close(fig)


def storage_threshold_iprob(ip=0.8):
    thresholds = [5, 7, 10, 15]
    # data50 = [read_data_n_agent_perturbations(
    #     n=50, iter=12000, threshold=t, time=10000, iprob=ip)[:,-1] for t in thresholds]
    data100 = [read_data_n_agent_perturbations(
        n=100, iter=12000, threshold=t, time=10000, iprob=ip)[:,-1] for t in thresholds]

    fig = plt.figure(figsize=(8,6), dpi=200)
    ax1 = fig.add_subplot(1, 1, 1)
    colordict = {
        0: 'gold',
        1: 'linen',
        2: 'orchid',
        3: 'peru',
        4: 'olivedrab',
        5: 'indianred',
        6: 'tomato'}
    colorshade = [
        'springgreen', 'lightcoral',
        'khaki', 'lightsalmon', 'deepskyblue']

    # labels = [ "> n/"+str(a) for a in thresholds]
    medianprops = dict(linewidth=2.5, color='firebrick')
    meanprops = dict(linewidth=2.5, color='#ff7f0e')
    positions = [
        [1, 2], [4, 5], [7, 8], [10, 11], # [13, 14]
        ]
    # datas = [
    #     [data50[0], data100[0]],
    #     [data50[1], data100[1]],
    #     [data50[2], data100[2]],
    #     [data50[3], data100[3]],
    #     # [data50[4], data100[4]],
    #     # [np.zeros(np.shape(datas50[4])), data100[5]],
    # ]

    # for j in range(len(positions)):
    bp1 = ax1.boxplot(
        data100, 0, 'gD', showmeans=True, meanline=True,
        patch_artist=True, medianprops=medianprops,
        meanprops=meanprops, widths=0.8)
    for patch, color in zip(bp1['boxes'], colordict.values()):
        patch.set_facecolor(color)

    ax1.legend(zip(bp1['boxes']), thresholds, fontsize="small", loc="upper center", title='Agent Population')
    # ax1.legend(zip(bp1['boxes']), thresholds, fontsize="small", loc="upper right", title='Storage Threshold')
    # ax1.set_xticks([1.5, 4.5, 7.5, 10.5])
    ax1.set_xticklabels(thresholds)
    ax1.set_yticks(range(0, 105, 20))
    ax1.set_xlabel('Storage Threshold', fontsize="large")
    ax1.set_ylabel('Foraging (%)', fontsize="large")
    plt.tight_layout()
    plt.title('IP('+str(ip)+')')
    maindir = '/tmp/swarm/data/experiments'
    fname = 'thresholdboxplotiprob'+str(ip)

    fig.savefig(
        maindir + '/' + fname + '.png')
    # pylint: disable = E1101

    plt.close(fig)


def storage_threshold_iprob_lt(ip=0.8):
    thresholds = [5, 7, 10, 15]
    # data50 = [read_data_n_agent_perturbations(
    #     n=50, iter=12000, threshold=t, time=10000, iprob=ip, idx=4)[:,-1] for t in thresholds]
    data100 = [read_data_n_agent_perturbations(
        n=100, iter=12000, threshold=t, time=10000, iprob=ip,idx=4)[:,-1] for t in thresholds]

    fig = plt.figure(figsize=(8,6), dpi=200)
    ax1 = fig.add_subplot(1, 1, 1)
    colordict = {
        0: 'gold',
        1: 'linen',
        2: 'orchid',
        3: 'peru',
        4: 'olivedrab',
        5: 'indianred',
        6: 'tomato'}
    colorshade = [
        'springgreen', 'lightcoral',
        'khaki', 'lightsalmon', 'deepskyblue']

    # labels = [ "> n/"+str(a) for a in thresholds]
    medianprops = dict(linewidth=2.5, color='firebrick')
    meanprops = dict(linewidth=2.5, color='#ff7f0e')
    positions = [
        [1, 2], [4, 5], [7, 8], [10, 11], # [13, 14]
        ]
    # datas = [
    #     [data50[0], data100[0]],
    #     [data50[1], data100[1]],
    #     [data50[2], data100[2]],
    #     [data50[3], data100[3]],
    #     # [data50[4], data100[4]],
    #     # [np.zeros(np.shape(datas50[4])), data100[5]],
    # ]

    # for j in range(len(positions)):
    bp1 = ax1.boxplot(
        data100, 0, 'gD', showmeans=True, meanline=True,
        patch_artist=True, medianprops=medianprops,
        meanprops=meanprops, widths=0.8)
    for patch, color in zip(bp1['boxes'], colordict.values()):
        patch.set_facecolor(color)

    ax1.legend(zip(bp1['boxes']), thresholds, fontsize="small", loc="upper center", title='Agent Population')
    # ax1.legend(zip(bp1['boxes']), thresholds, fontsize="small", loc="upper right", title='Storage Threshold')
    # ax1.set_xticks([1.5, 4.5, 7.5, 10.5])
    ax1.set_xticklabels(thresholds)
    ax1.set_yticks(range(0, 20, 2))
    ax1.set_xlabel('Storage Threshold', fontsize="large")
    ax1.set_ylabel('Average LT Rate', fontsize="large")
    plt.tight_layout()
    plt.title('IP('+str(ip)+')')
    maindir = '/tmp/swarm/data/experiments'
    fname = 'thresholdboxplotltrate'+str(ip)

    fig.savefig(
        maindir + '/' + fname + '.png')
    # pylint: disable = E1101

    plt.close(fig)


def storage_threshold_iprob_gs(ip=0.8):
    thresholds = [5, 7, 10, 15]
    # data50 = [read_data_n_agent_perturbations(
    #     n=50, iter=12000, threshold=t, time=10000, iprob=ip, idx=6)[:,-1] for t in thresholds]
    data100 = [read_data_n_agent_perturbations(
        n=100, iter=12000, threshold=t, time=10000, iprob=ip,idx=6)[:,-1] for t in thresholds]

    fig = plt.figure(figsize=(8,6), dpi=200)
    ax1 = fig.add_subplot(1, 1, 1)
    colordict = {
        0: 'gold',
        1: 'linen',
        2: 'orchid',
        3: 'peru',
        4: 'olivedrab',
        5: 'indianred',
        6: 'tomato'}
    colorshade = [
        'springgreen', 'lightcoral',
        'khaki', 'lightsalmon', 'deepskyblue']

    # labels = [ "> n/"+str(a) for a in thresholds]
    medianprops = dict(linewidth=2.5, color='firebrick')
    meanprops = dict(linewidth=2.5, color='#ff7f0e')
    positions = [
        [1, 2], [4, 5], [7, 8], [10, 11], # [13, 14]
        ]
    # datas = [
    #     [data50[0], data100[0]],
    #     [data50[1], data100[1]],
    #     [data50[2], data100[2]],
    #     [data50[3], data100[3]],
    #     # [data50[4], data100[4]],
    #     # [np.zeros(np.shape(datas50[4])), data100[5]],
    # ]

    # for j in range(len(positions)):
    bp1 = ax1.boxplot(
        data100, 0, 'gD', showmeans=True, meanline=True,
        patch_artist=True, medianprops=medianprops,
        meanprops=meanprops, widths=0.8)
    for patch, color in zip(bp1['boxes'], colordict.values()):
        patch.set_facecolor(color)

    ax1.legend(zip(bp1['boxes']), thresholds, fontsize="small", loc="upper center", title='Agent Population')
    # ax1.legend(zip(bp1['boxes']), thresholds, fontsize="small", loc="upper right", title='Storage Threshold')
    # ax1.set_xticks([1.5, 4.5, 7.5, 10.5])
    ax1.set_xticklabels(thresholds)
    ax1.set_yticks(range(0, 80, 10))
    ax1.set_xlabel('Storage Threshold', fontsize="large")
    ax1.set_ylabel('Genetic Step Rate', fontsize="large")
    plt.title('IP('+str(ip)+')')
    plt.tight_layout()
    maindir = '/tmp/swarm/data/experiments'
    fname = 'thresholdboxplotgeneticstep'+str(ip)

    fig.savefig(
        maindir + '/' + fname + '.png')
    # pylint: disable = E1101

    plt.close(fig)


def storage_threshold_all_100(ip=0.8):
    thresholds = [5, 7, 10, 15]
    # data50 = [read_data_n_agent_perturbations(
    #     n=50, iter=12000, threshold=t, time=10000, iprob=ip, idx=6)[:,-1] for t in thresholds]
    data100 = [read_data_n_agent_perturbations_all(
        n=100, iter=12000, threshold=t, time=10000, iprob=ip,idx=[2,4,6])[:,:,-1] for t in thresholds]

    # print(data100[0].shape)
    fig = plt.figure(figsize=(8,6), dpi=200)
    ax1 = fig.add_subplot(1, 1, 1)
    colordict = {
        0: 'gold',
        1: 'linen',
        2: 'orchid',
        3: 'peru',
        4: 'olivedrab',
        5: 'indianred',
        6: 'tomato'}
    colorshade = [
        'springgreen', 'lightcoral',
        'khaki', 'lightsalmon', 'deepskyblue']

    # labels = [ "> n/"+str(a) for a in thresholds]
    medianprops = dict(linewidth=2.5, color='firebrick')
    meanprops = dict(linewidth=2.5, color='#ff7f0e')
    positions = [
        [1, 2], [5,6], [9,10], [13, 14], # [13, 14]
        ]
    datas = [
        [data100[0][0,:],data100[0][2,:]],
        [data100[1][0,:],data100[1][2,:]],
        [data100[2][0,:],data100[2][2,:]],
        [data100[3][0,:],data100[3][2,:]],
    ]

    for j in range(len(positions)):
        bp1 = ax1.boxplot(
            datas[j], 0, 'gD', showmeans=True, meanline=True,
            patch_artist=True, medianprops=medianprops,
            meanprops=meanprops, widths=0.8, positions=positions[j])
        for patch, color in zip(bp1['boxes'], colordict.values()):
            patch.set_facecolor(color)

    ax1.legend(zip(bp1['boxes']), ['Foraging', 'Genetic Step'], fontsize="small", loc="upper right", title='Metrices')
    # ax1.legend(zip(bp1['boxes']), thresholds, fontsize="small", loc="upper right", title='Storage Threshold')
    ax1.set_xticks([2, 6, 10, 14])
    ax1.set_xticklabels(thresholds)
    ax1.set_yticks(range(0, 105, 10))
    ax1.set_xlabel('Storage Threshold', fontsize="large")
    ax1.set_ylabel('Performance', fontsize="large")

    ax2 = ax1.twinx()
    datas = [data100[0][1,:],
            data100[1][1,:],
            data100[2][1,:],
            data100[3][1,:]
            ]
    bp2 = ax2.boxplot(
        datas, 0, 'gD', showmeans=True, meanline=True,
        patch_artist=True, medianprops=medianprops,
        meanprops=meanprops, widths=0.8, positions=[3,7,11,15])
    for patch in bp2['boxes']:
        patch.set_facecolor(list(colordict.values())[2])

    # ax2.set_xticks([2, 6, 10, 14])
    # ax2.set_xticklabels(thresholds)
    ax2.legend(zip(bp2['boxes']), ['Lateral Transfer'], fontsize="small", loc="upper center")
    ax2.set_yticks(range(0, 20, 2))
    ax2.set_xlabel('Storage Threshold', fontsize="large")
    ax2.set_ylabel('LT Rate', fontsize="large")

    plt.title('IP('+str(ip)+')')
    plt.tight_layout()
    maindir = '/tmp/swarm/data/experiments'
    fname = 'thresholdboxplotall'+str(ip)

    fig.savefig(
        maindir + '/' + fname + '.png')
    # pylint: disable = E1101

    plt.close(fig)


def obstacle_introduced():
    timings = range(1000, 11001, 1000)
    # data50 = [read_data_n_agent_perturbations(
    #     n=50, iter=12000, threshold=t, time=10000, iprob=ip, idx=6)[:,-1] for t in thresholds]
    data = [np.squeeze(read_data_n_agent_perturbations_all(
        n=100, iter=12000, threshold=7, time=t, iprob=0.85,
        addobject='Obstacles',no_objects=5, radius=10)[:,:,-1]) for t in timings]

    fig = plt.figure(figsize=(8,6), dpi=200)
    ax1 = fig.add_subplot(1, 1, 1)
    colordict = {
        0: 'gold',
        1: 'linen',
        2: 'orchid',
        3: 'peru',
        4: 'olivedrab',
        5: 'indianred',
        6: 'tomato'}
    colorshade = [
        'springgreen', 'lightcoral',
        'khaki', 'lightsalmon', 'deepskyblue']

    # labels = [ "> n/"+str(a) for a in thresholds]
    medianprops = dict(linewidth=2.5, color='firebrick')
    meanprops = dict(linewidth=2.5, color='#ff7f0e')
    # positions = [
    #     [1, 2], [5,6], [9,10], [13, 14], # [13, 14]
    #     ]

    # for j in range(len(positions)):
    bp1 = ax1.boxplot(
        data, 0, 'gD', showmeans=True, meanline=True,
        patch_artist=True, medianprops=medianprops,
        meanprops=meanprops, widths=0.8)
    for patch, color in zip(bp1['boxes'], colordict.values()):
        patch.set_facecolor(color)

    # ax1.legend(zip(bp1['boxes']), ['Foraging', 'Genetic Step'], fontsize="small", loc="upper right", title='Metrices')
    ax1.legend(zip(bp1['boxes']), timings, fontsize="small", loc="lower left", title='Perturbation Timings')
    # ax1.set_xticks(timi)
    ax1.set_xticklabels(timings)
    ax1.set_yticks(range(0, 105, 10))
    ax1.set_xlabel('Steps', fontsize="large")
    ax1.set_ylabel('Foraging (%)', fontsize="large")

    # plt.title('IP('+str(ip)+')')
    plt.tight_layout()
    maindir = '/tmp/swarm/data/experiments'
    fname = 'obstacle_introduced'

    fig.savefig(
        maindir + '/' + fname + '.png')
    # pylint: disable = E1101

    plt.close(fig)


def trap_introduced():
    timings = range(1000, 6001, 1000)
    # data50 = [read_data_n_agent_perturbations(
    #     n=50, iter=12000, threshold=t, time=10000, iprob=ip, idx=6)[:,-1] for t in thresholds]
    data = [np.squeeze(read_data_n_agent_perturbations_all(
        n=100, iter=12000, threshold=7, time=t, iprob=0.85,
        addobject='Traps',no_objects=5, radius=10, idx=[2,3])[:,:,-1]) for t in timings]
    forgedata = [data[i][0,:] for i in range(len(data))]
    deaddata = [data[i][1,:] for i in range(len(data))]
    fig = plt.figure(figsize=(12,5), dpi=200)
    ax1 = fig.add_subplot(1, 2, 1)
    colordict = {
        0: 'gold',
        1: 'linen',
        2: 'orchid',
        3: 'peru',
        4: 'olivedrab',
        5: 'indianred',
        6: 'tomato'}
    colorshade = [
        'springgreen', 'lightcoral',
        'khaki', 'lightsalmon', 'deepskyblue']

    # labels = [ "> n/"+str(a) for a in thresholds]
    medianprops = dict(linewidth=2.5, color='firebrick')
    meanprops = dict(linewidth=2.5, color='#ff7f0e')
    # positions = [
    #     [1, 2], [5,6], [9,10], [13, 14], # [13, 14]
    #     ]

    # for j in range(len(positions)):
    bp1 = ax1.boxplot(
        forgedata, 0, 'gD', showmeans=True, meanline=True,
        patch_artist=True, medianprops=medianprops,
        meanprops=meanprops, widths=0.8)
    for patch, color in zip(bp1['boxes'], colordict.values()):
        patch.set_facecolor(color)

    # ax1.legend(zip(bp1['boxes']), ['Foraging', 'Genetic Step'], fontsize="small", loc="upper right", title='Metrices')
    ax1.legend(zip(bp1['boxes']), timings, fontsize="small", loc="upper left", title='Perturbation Timings')
    # ax1.set_xticks(timi)
    ax1.set_xticklabels(timings)
    ax1.set_yticks(range(0, 105, 10))
    ax1.set_xlabel('Perturbation Timings', fontsize="large")
    ax1.set_ylabel('Foraging (%)', fontsize="large")

    ax2 = fig.add_subplot(1, 2, 2)
    bp2 = ax2.boxplot(
        deaddata, 0, 'gD', showmeans=True, meanline=True,
        patch_artist=True, medianprops=medianprops,
        meanprops=meanprops, widths=0.8)
    for patch, color in zip(bp2['boxes'], colordict.values()):
        patch.set_facecolor(color)

    # ax1.legend(zip(bp1['boxes']), ['Foraging', 'Genetic Step'], fontsize="small", loc="upper right", title='Metrices')
    ax2.legend(zip(bp1['boxes']), timings, fontsize="small", loc="lower left", title='Perturbation Timings')
    # ax1.set_xticks(timi)
    ax2.set_xticklabels(timings)
    ax2.set_yticks(range(0, 105, 10))
    ax2.set_xlabel('Perturbation Timings', fontsize="large")
    ax2.set_ylabel('Dead Agents', fontsize="large")
    # plt.title('IP('+str(ip)+')')
    plt.tight_layout()
    maindir = '/tmp/swarm/data/experiments'
    fname = 'traps_introduced'

    fig.savefig(
        maindir + '/' + fname + '.png')
    # pylint: disable = E1101

    plt.close(fig)


def trap_introduced_deadagent():
    timings = range(1000, 6001, 1000)
    # data50 = [read_data_n_agent_perturbations(
    #     n=50, iter=12000, threshold=t, time=10000, iprob=ip, idx=6)[:,-1] for t in thresholds]
    data = [np.mean(np.squeeze(read_data_n_agent_perturbations_all(
        n=100, iter=12000, threshold=7, time=t, iprob=0.85,
        addobject='Traps',no_objects=5, radius=10, idx=[3])), axis=0) for t in timings]
    # data = [np.mean(data[i], axis=0) for i in range(len(data))]
    fig = plt.figure(figsize=(8,6), dpi=200)
    ax1 = fig.add_subplot(1, 1, 1)

    xvalues = range(12002)
    for i in range(len(timings)):
        ax1.scatter(xvalues, data[i], label=str(timings[i]), alpha=0.2, s=5)
    # ax1.legend(zip(bp1['boxes']), ['Foraging', 'Genetic Step'], fontsize="small", loc="upper right", title='Metrices')
    ax1.legend(fontsize="small", loc="upper left", title='Perturbation Timings')
    # ax1.set_xticks(timi)
    # ax1.set_xticklabels(timings)
    ax1.set_yticks(range(0, 105, 10))
    ax1.set_xlabel('Perturbation Timings', fontsize="large")
    ax1.set_ylabel('No. of Dead Agents', fontsize="large")

    # plt.title('IP('+str(ip)+')')
    plt.tight_layout()
    maindir = '/tmp/swarm/data/experiments'
    fname = 'traps_introduced_deadagent_perstep'

    fig.savefig(
        maindir + '/' + fname + '.png')
    # pylint: disable = E1101

    plt.close(fig)


def plot_lt_foraging_gentic(ip=0.85, time=1000):
    fig = plt.figure(figsize=(8,6), dpi=200)
    data = np.mean(np.squeeze(read_data_n_agent_perturbations_all(
        n=100, iter=12000, threshold=7, time=time, iprob=0.85,
        addobject='Obstacles',no_objects=5, radius=10, idx=[2,4,6])), axis=1)

    # data = read_data_n_agent_perturbations_all(
    #     n=100, iter=12000, threshold=5, time=time, iprob=ip,idx=[2,4,6])
    # print(data[0,:].shape)
    # exit()
    # obstaclestime = [read_data_n_agent_perturbations(addobject='Obstacles', no_objects=5, radius=10, time=t) for t in time]
    ax1 = fig.add_subplot(1, 1, 1)
    xvalues = np.array(list(range(12002)))
    mask = xvalues % 500 == 0
    ax1.plot(xvalues[mask], data[1,:][mask], 'o-', label='Lateral Transfer Rate')
    ax1.plot(xvalues[mask], data[2,:][mask], 'o-', label='Genetic Step Rate')
    ax1.legend(fontsize="small", loc="upper left", title='Metric')
    ax1.set_xlabel('Evolution Steps')
    ax1.set_ylabel('Rate per step')
    ax1.set_yticks(range(0, 50, 10))

    ax2 = ax1.twinx()
    ax2.plot(xvalues[mask], data[0,:][mask], 'o-', label='Foraging', color='green')
    # traps_mean = np.mean(traps, axis=1)
    ax2.legend(fontsize="small", loc="upper right")

    # ax1.set_xticklabels(['Foraging', 'Maintenance'])
    ax2.set_xlabel('Evolution Steps')
    ax2.set_ylabel('Foraging %')
    ax2.set_yticks(range(0, 105, 20))
    plt.title('Timing('+str(time)+')')
    plt.tight_layout()
    maindir = '/tmp/swarm/data/experiments/'
    # fname = 'agentsitecomp' + agent
    nadir = os.path.join(maindir, str(100))

    fig.savefig(
        nadir + 'thresholdallplot' +str(ip) +'_' +str(time)+ '.png')
    plt.close(fig)


def plot_foraging_baseline(ip=0.85, time=10000):
    fig = plt.figure(figsize=(8,6), dpi=200)
    # data = np.mean(np.squeeze(read_data_n_agent_perturbations_all(
    #     n=100, iter=12000, threshold=7, time=time, iprob=0.85,
    #     no_objects=1, radius=5, idx=[2,4,6])), axis=1)

    data = np.squeeze(read_data_n_agent_perturbations_all(
        n=100, iter=12000, threshold=7, time=time, iprob=0.85,
        no_objects=1, radius=5, idx=[2]))
    print(data.shape)
    q2 = np.median(data, axis=0)
    q1 = np.quantile(data, axis=0, q=0.25)
    q3 = np.quantile(data, axis=0, q=0.75)
    print(q2.shape, q1.shape, q3.shape)
    ax1 = fig.add_subplot(1, 1, 1)
    xvalues = np.array(list(range(12002)))
    # mask = xvalues % 500 == 0
    mask = [True] * len(xvalues)
    ax1.plot(xvalues[mask], q2[mask], '-', label='Foraging', color='green')
    ax1.fill_between(
                xvalues[mask], q1[mask], q3[mask], color='seagreen', alpha=0.3)

    colordict = {
            0: 'gold',
            1: 'linen',
            2: 'orchid',
            3: 'peru',
            4: 'olivedrab',
            5: 'indianred',
            6: 'tomato'}
    colorshade = [
        'springgreen', 'lightcoral',
        'khaki', 'lightsalmon', 'deepskyblue']

    medianprops = dict(linewidth=2.5, color='firebrick')
    meanprops = dict(linewidth=2.5, color='#ff7f0e')

    bp1 = ax1.boxplot(
        [data[:,-1]], 0, 'gD', showmeans=True, meanline=True,
        patch_artist=True, medianprops=medianprops,
        meanprops=meanprops, widths=500, positions=[12000])
    for patch, color in zip(bp1['boxes'], colordict.values()):
        patch.set_facecolor(color)

    # traps_mean = np.mean(traps, axis=1)
    ax1.legend(fontsize="small", loc="upper left", title="Learning Efficiency")
    # ax1.set_xticklabels(['Foraging', 'Maintenance'])
    ax1.set_xlabel('Evolution Steps')
    ax1.set_xlim(0, 13000)
    ax1.set_xticks(range(0,13001,1000))
    ax1.set_xticklabels(range(0,13001,1000))
    ax1.set_ylabel('Foraging (%)')
    ax1.set_yticks(range(0, 105, 20))
    ax2 = ax1.twinx()
    ax2.set_yticks(range(0, 105, 20))

    plt.title('Baseline')
    plt.tight_layout()
    maindir = '/tmp/swarm/data/experiments/'
    fig.savefig(
        maindir + 'baselineallplot'+ '.png')
    plt.close(fig)


def plot_foraging_baseline_trap(basedata, ip=0.85, time=1000):
    fig = plt.figure(figsize=(8,6), dpi=200)
    axeslimitdict = {
        1000:[(1000,2000), (0,8)],
        2000:[(2000,3000), (4,15)],
        3000:[(3000,4000), (10,30)],
        4000:[(4000,5000), (20,50)],
        5000:[(5000,6000), (30,55)],
        6000:[(6000,7000), (40,60)],
        7000:[(7000,8000), (50,70)],
        8000:[(8000,9000), (60,80)],
        9000:[(9000,10000), (70,90)],
        10000:[(10000,11000), (75,90)],
        11000:[(11000,12000), (75,95)]
        }
    # basedata = np.squeeze(read_data_n_agent_perturbations_all(
    #     n=100, iter=12000, threshold=7, time=10000, iprob=0.85,
    #     no_objects=1, radius=5, idx=[2,4]))

    baseq2 = np.median(basedata, axis=1)
    baseq1 = np.quantile(basedata, axis=1, q=0.25)
    baseq3 = np.quantile(basedata, axis=1, q=0.75)
    ax1 = fig.add_subplot(1, 1, 1)
    xvalues = np.array(list(range(12002)))
    # mask = xvalues % 500 == 0
    mask = [True] * len(xvalues)
    ax1.plot(xvalues[mask], baseq2[0,:][mask], '-', label='Baseline Foraging', color='green')
    ax1.fill_between(
                xvalues[mask], baseq1[0,:][mask], baseq3[0,:][mask], color='seagreen', alpha=0.3)

    ax_zoom = ax1.inset_axes([0.05,0.5,0.47,0.47], alpha=0.3)
    data = np.squeeze(read_data_n_agent_perturbations_all(
        n=100, iter=12000, threshold=7, time=time, iprob=0.85, addobject='Traps',
        no_objects=1, radius=10, idx=[2,3]))

    q2 = np.median(data, axis=1)
    q1 = np.quantile(data, axis=1, q=0.25)
    q3 = np.quantile(data, axis=1, q=0.75)
    xvalues = np.array(list(range(12002)))
    mask = [True] * len(xvalues)
    ax1.plot(xvalues[mask], q2[0,:][mask], '-', label='Foraging with Trap', color='red')
    ax1.fill_between(
                xvalues[mask], q1[0,:][mask], q3[0,:][mask], color='salmon', alpha=0.3)

    ax_zoom.plot(xvalues[mask], q2[0,:][mask], '-', label='Trap', color='red')
    ax_zoom.plot(xvalues[mask], baseq2[0,:][mask], '-', label='Baseline', color='green')
    ax_zoom.set_xlim(axeslimitdict[time][0][0],axeslimitdict[time][0][1])
    ax_zoom.set_ylim(axeslimitdict[time][1][0],axeslimitdict[time][1][1])
    ax1.indicate_inset_zoom(ax_zoom, edgecolor="black", label="Zoomed", alpha=0.3)

    ax1.legend(fontsize="small", loc="lower right", title="Learning Efficiency")
    ax1.set_xlabel('Evolution Steps')
    ax1.set_ylabel('Foraging (%)')
    ax1.set_yticks(range(0, 105, 20))
    # ax2 = ax1.twinx()
    # ax2.set_yticks(range(0, 105, 20))

    plt.title('Trap Added at ' + str(time) +' Step')
    plt.tight_layout()
    maindir = '/tmp/swarm/data/experiments/'
    # fname = 'agentsitecomp' + agent
    # nadir = os.path.join(maindir, str(100))

    fig.savefig(
        maindir + 'baselinetrap_zoom_'+ str(time)+ '.png')
    plt.close(fig)


def plot_foraging_baseline_trap_deadagent(basedata, ip=0.85, time=1000):
    fig = plt.figure(figsize=(8,6), dpi=200)
    axeslimitdict = {
        1000:[(1000,2000), (0,8)],
        2000:[(2000,3000), (0,15)],
        3000:[(3000,4000), (0,20)],
        4000:[(4000,5000), (0,20)],
        5000:[(5000,6000), (0,20)],
        6000:[(6000,7000), (0,20)],
        7000:[(7000,8000), (0,25)],
        8000:[(8000,9000), (0,25)],
        9000:[(9000,10000), (0,30)],
        10000:[(10000,11000), (0,35)],
        11000:[(11000,12000), (0,35)]
        }
    # basedata = np.squeeze(read_data_n_agent_perturbations_all(
    #     n=100, iter=12000, threshold=7, time=10000, iprob=0.85,
    #     no_objects=1, radius=5, idx=[2,3]))

    baseq2 = np.median(basedata, axis=1)
    baseq1 = np.quantile(basedata, axis=1, q=0.25)
    baseq3 = np.quantile(basedata, axis=1, q=0.75)
    ax1 = fig.add_subplot(1, 1, 1)
    xvalues = np.array(list(range(12002)))
    # mask = xvalues % 500 == 0
    mask = [True] * len(xvalues)
    ax1.plot(xvalues[mask], baseq2[1,:][mask], '-', label='Baseline Deadagents', color='orange')
    ax1.fill_between(
                xvalues[mask], baseq1[1,:][mask], baseq3[1,:][mask], color='peachpuff', alpha=0.3)

    ax_zoom = ax1.inset_axes([0.05,0.5,0.47,0.47], alpha=0.3)
    data = np.squeeze(read_data_n_agent_perturbations_all(
        n=100, iter=12000, threshold=7, time=time, iprob=0.85, addobject='Traps',
        no_objects=1, radius=10, idx=[2,3]))

    q2 = np.median(data, axis=1)
    q1 = np.quantile(data, axis=1, q=0.25)
    q3 = np.quantile(data, axis=1, q=0.75)
    xvalues = np.array(list(range(12002)))
    mask = [True] * len(xvalues)
    ax1.plot(xvalues[mask], q2[1,:][mask], '-', label='Dead agents with Trap Addition', color='red')
    ax1.fill_between(
                xvalues[mask], q1[1,:][mask], q3[1,:][mask], color='salmon', alpha=0.3)

    ax_zoom.plot(xvalues[mask], q2[1,:][mask], '-', label='Trap', color='red')
    ax_zoom.plot(xvalues[mask], baseq2[1,:][mask], '-', label='Baseline', color='orange')
    ax_zoom.set_xlim(axeslimitdict[time][0][0],axeslimitdict[time][0][1])
    ax_zoom.set_ylim(axeslimitdict[time][1][0],axeslimitdict[time][1][1])
    ax1.indicate_inset_zoom(ax_zoom, edgecolor="black", label="Zoomed", alpha=0.3)
    ax1.set_xlabel('Evolution Steps')
    ax1.set_ylabel('Dead Agents (%) / Foraging(%)')
    # ax2 = ax1.twinx()
    ax1.plot(xvalues[mask], baseq2[0,:][mask], '-', label='Baseline Foraging', color='green')
    ax1.plot(xvalues[mask], q2[0,:][mask], '--', label='Pertubed Foraging', color='seagreen')
    ax1.set_yticks(range(0, 105, 20))
    # ax2.set_yticks(range(0, 105, 20))
    # ax2.set_ylabel('Foraging (%)')
    ax1.legend(fontsize="small", loc="center right", title="Metrics")
    plt.title('Trap Added at ' + str(time) +' Step')
    plt.tight_layout()
    maindir = '/tmp/swarm/data/experiments/'
    # fname = 'agentsitecomp' + agent
    # nadir = os.path.join(maindir, str(100))

    fig.savefig(
        maindir + 'baselinetrap_deadagents_zoom_'+ str(time)+ '.png')
    plt.close(fig)



def plot_foraging_deadagent_curve(basedata, ip=0.85):
    fig = plt.figure(figsize=(8,6), dpi=200)

    ax1 = fig.add_subplot(1, 1, 1)
    xvalues = np.array(list(range(12002)))
    # mask = xvalues % 500 == 0
    mask = [True] * len(xvalues)

    data = [np.squeeze(read_data_n_agent_perturbations_all(
        n=100, iter=12000, threshold=7, time=time, iprob=0.85, addobject='Traps',
        no_objects=1, radius=10, idx=[3]))[:,-1] for time in range(1000,11001, 1000)]
    # data = [basedata] + data
    mask = [True] * len(xvalues)
    colordict = {
        0: 'gold',
        1: 'linen',
        2: 'orchid',
        3: 'peru',
        4: 'olivedrab',
        5: 'indianred',
        6: 'tomato'}
    colorshade = [
        'springgreen', 'lightcoral',
        'khaki', 'lightsalmon', 'deepskyblue']

    medianprops = dict(linewidth=2.5, color='firebrick')
    meanprops = dict(linewidth=2.5, color='#ff7f0e')

    bp1 = ax1.boxplot(
        data, 0, 'gD', showmeans=True, meanline=True,
        patch_artist=True, medianprops=medianprops,
        meanprops=meanprops, widths=0.8)
    for patch, color in zip(bp1['boxes'], colordict.values()):
        patch.set_facecolor(color)

    ax1.set_xlabel('Trap Addition Time')
    ax1.set_ylabel('Dead Agents (%)')
    ax1.set_yticks(range(0, 70, 10))
    # ax1.set_xticks( range(0,11001,1000))
    ax1.set_xticklabels(range(1000,11001,1000))
    ax1.legend(zip(bp1['boxes']), range(1000,11001, 1000), fontsize="small", loc="upper right", title='Trap Addition Time')
    # ax1.legend(fontsize="small", loc="center right", title="Trap Addition time")
    # plt.title('Trap Added at ' + str(time) +' Step')
    plt.tight_layout()
    maindir = '/tmp/swarm/data/experiments/'
    # fname = 'agentsitecomp' + agent
    # nadir = os.path.join(maindir, str(100))

    fig.savefig(
        maindir + 'baselinetrap_trap_deadagents_boxplot_all.png')
    plt.close(fig)


def plot_foraging_baseline_obstacles(basedata, ip=0.85, time=1000):
    fig = plt.figure(figsize=(8,6), dpi=200)
    axeslimitdict = {
        1000:[(1000,2000), (0,8)],
        2000:[(2000,3000), (4,15)],
        3000:[(3000,4000), (10,30)],
        4000:[(4000,5000), (20,50)],
        5000:[(5000,6000), (30,55)],
        6000:[(6000,7000), (40,60)],
        7000:[(7000,8000), (50,70)],
        8000:[(8000,9000), (60,80)],
        9000:[(9000,10000), (70,90)],
        10000:[(10000,11000), (75,90)],
        11000:[(11000,12000), (75,95)]
        }
    # basedata = np.squeeze(read_data_n_agent_perturbations_all(
    #     n=100, iter=12000, threshold=7, time=10000, iprob=0.85,
    #     no_objects=1, radius=5, idx=[2,4]))

    baseq2 = np.median(basedata, axis=1)
    baseq1 = np.quantile(basedata, axis=1, q=0.25)
    baseq3 = np.quantile(basedata, axis=1, q=0.75)
    ax1 = fig.add_subplot(1, 1, 1)
    xvalues = np.array(list(range(12002)))
    # mask = xvalues % 500 == 0
    mask = [True] * len(xvalues)
    ax1.plot(xvalues[mask], baseq2[0,:][mask], '-', label='Baseline Foraging', color='green')
    ax1.fill_between(
                xvalues[mask], baseq1[0,:][mask], baseq3[0,:][mask], color='seagreen', alpha=0.3)

    ax_zoom = ax1.inset_axes([0.05,0.5,0.47,0.47], alpha=0.3)
    data = np.squeeze(read_data_n_agent_perturbations_all(
        n=100, iter=12000, threshold=7, time=time, iprob=0.85, addobject='Obstacles',
        no_objects=2, radius=10, idx=[2,3]))

    q2 = np.median(data, axis=1)
    q1 = np.quantile(data, axis=1, q=0.25)
    q3 = np.quantile(data, axis=1, q=0.75)
    xvalues = np.array(list(range(12002)))
    mask = [True] * len(xvalues)
    ax1.plot(xvalues[mask], q2[0,:][mask], '-', label='Foraging with Obstacle', color='red')
    ax1.fill_between(
                xvalues[mask], q1[0,:][mask], q3[0,:][mask], color='salmon', alpha=0.3)

    ax_zoom.plot(xvalues[mask], q2[0,:][mask], '-', label='Obstacle', color='red')
    ax_zoom.plot(xvalues[mask], baseq2[0,:][mask], '-', label='Baseline', color='green')
    ax_zoom.set_xlim(axeslimitdict[time][0][0],axeslimitdict[time][0][1])
    ax_zoom.set_ylim(axeslimitdict[time][1][0],axeslimitdict[time][1][1])
    ax1.indicate_inset_zoom(ax_zoom, edgecolor="black", label="Zoomed", alpha=0.3)

    ax1.legend(fontsize="small", loc="lower right", title="Learning Efficiency")
    ax1.set_xlabel('Evolution Steps')
    ax1.set_ylabel('Foraging (%)')
    ax1.set_yticks(range(0, 105, 20))
    # ax2 = ax1.twinx()
    # ax2.set_yticks(range(0, 105, 20))

    plt.title('Obstacle Added at ' + str(time) +' Step')
    plt.tight_layout()
    maindir = '/tmp/swarm/data/experiments/'
    # fname = 'agentsitecomp' + agent
    # nadir = os.path.join(maindir, str(100))

    fig.savefig(
        maindir + 'baseline_obstacle_zoom_'+ str(time)+ '.png')
    plt.close(fig)


def plot_no_obstacles_performance(ip=0.85, time=1000):
    fig = plt.figure(figsize=(8,6), dpi=200)
    ax1 = fig.add_subplot(1, 1, 1)

    data = [np.squeeze(read_data_n_agent_perturbations_all(
        n=100, iter=12000, threshold=7, time=time, iprob=ip, addobject='Obstacles',
        no_objects=n, radius=10, idx=[2]))[:,-1] for n in range(1,6)]

    colordict = {
        0: 'gold',
        1: 'linen',
        2: 'orchid',
        3: 'peru',
        4: 'olivedrab',
        5: 'indianred',
        6: 'tomato'}
    colorshade = [
        'springgreen', 'lightcoral',
        'khaki', 'lightsalmon', 'deepskyblue']

    medianprops = dict(linewidth=2.5, color='firebrick')
    meanprops = dict(linewidth=2.5, color='#ff7f0e')

    bp1 = ax1.boxplot(
        data, 0, 'gD', showmeans=True, meanline=True,
        patch_artist=True, medianprops=medianprops,
        meanprops=meanprops, widths=0.8)
    for patch, color in zip(bp1['boxes'], colordict.values()):
        patch.set_facecolor(color)

    ax1.set_xlabel('No. of Obstacles')
    ax1.set_ylabel('Foraging (%)')
    ax1.set_yticks(range(0, 105, 10))
    # ax1.set_xticks( range(0,11001,1000))
    # ax1.set_xticklabels(range(1000,11001,1000))
    ax1.legend(zip(bp1['boxes']), range(1,6), fontsize="small", loc="lower left", title='No. of Obstacles')
    plt.title('No. of Obstacle Added at ' + str(time) +' Step')
    plt.tight_layout()
    maindir = '/tmp/swarm/data/experiments/'

    fig.savefig(
        maindir + 'no_obstacle_'+ str(time)+ '.png')
    plt.close(fig)


def plot_shift_foraging_lt_on_off():
    fig = plt.figure(figsize=(8,6), dpi=200)
    ax1 = fig.add_subplot(1, 1, 1)
    timings = range(1000, 4001, 1000)
    data = [np.squeeze(read_data_n_agent_perturbations_all_shift(
        n=100, iter=12000, threshold=7, time=1000, iprob=0.85,
        addobject='Obstacles', no_objects=1,
        radius=10, idx=[2], ltstopint=t))[:,-1] for t in timings]
    baseline = [np.squeeze(read_data_n_agent_perturbations_all_shift(
        n=100, iter=12000, threshold=7, time=10000, iprob=0.85,
        addobject='None', no_objects=1,
        radius=5, idx=[2]))[:,-1]]
    data = baseline + data
    colordict = {
        0: 'gold',
        1: 'linen',
        2: 'orchid',
        3: 'peru',
        4: 'olivedrab',
        5: 'indianred',
        6: 'tomato'}
    colorshade = [
        'springgreen', 'lightcoral',
        'khaki', 'lightsalmon', 'deepskyblue']

    medianprops = dict(linewidth=2.5, color='firebrick')
    meanprops = dict(linewidth=2.5, color='#ff7f0e')

    bp1 = ax1.boxplot(
        data, 0, 'gD', showmeans=True, meanline=True,
        patch_artist=True, medianprops=medianprops,
        meanprops=meanprops, widths=0.8)
    for patch, color in zip(bp1['boxes'], colordict.values()):
        patch.set_facecolor(color)

    ax1.set_xlabel('LT Off Duration')
    ax1.set_ylabel('Foraging (%)')
    ax1.set_yticks(range(0, 105, 10))
    label = ['Baseline'] + list(range(1000, 4001, 1000))
    # ax1.set_xticks( )
    ax1.set_xticklabels(label)
    ax1.legend(zip(bp1['boxes']), label, fontsize="small", loc="lower left", title='LT Off Duration')
    # plt.title('No. of Obstacle Added at ' + str(time) +' Step')
    plt.tight_layout()
    maindir = '/tmp/swarm/data/experiments/'

    fig.savefig(
        maindir + 'lt_off_duration.png')
    plt.close(fig)


def obstacle_introduced_compare():
    timings = range(7000, 11001, 1000)
    data = [np.squeeze(read_data_n_agent_perturbations_all(
        n=100, iter=12000, threshold=7, time=1000, iprob=0.85,
        addobject='Obstacles', no_objects=1,
        radius=10, idx=[2], fname=fname, ltstopint=t)) for t in timings]

    fig = plt.figure(figsize=(8,6), dpi=200)
    ax1 = fig.add_subplot(1, 1, 1)


def ip_paper_efficiency_power(t=5):
    ip = [0.5, 0.7, 0.8, 0.85, 0.9, 0.99]
    axeslimitdict = {
        0:[(11500,12002), (75,95)]
        }
    data = [np.median(np.squeeze(read_data_n_agent_perturbations_all(
        n=100, iter=12000, threshold=t, time=10000, iprob=i,
        addobject='None',no_objects=1, radius=5, idx=[2])), axis=0) for i in ip]

    fig = plt.figure(figsize=(6,4), dpi=200)
    ax1 = fig.add_subplot(1, 1, 1)

    xvalues = np.array(list(range(12002)))
    mask = xvalues % 1000 == 0
    colors = []
    for i in range(len(ip)):
        p = ax1.plot(xvalues[mask], data[i][mask], marker="v", ls='-', label=str(ip[i]))
        colors += [p[0].get_color()]

    ax_zoom = ax1.inset_axes([0.08,0.5,0.47,0.47])
    ax_zoom.patch.set_alpha(0.5)

    data100 = [read_data_n_agent_perturbations(
        n=100, iter=12000, threshold=t, time=10000, iprob=i)[:,-1] for i in ip]
    colordict = {
        0: 'gold',
        1: 'linen',
        2: 'orchid',
        3: 'peru',
        4: 'olivedrab',
        5: 'indianred',
        6: 'tomato'}
    colorshade = [
        'springgreen', 'lightcoral',
        'khaki', 'lightsalmon', 'deepskyblue']

    medianprops = dict(linewidth=2.5, color='firebrick')
    meanprops = dict(linewidth=2.5, color='#ff7f0e')
    bp1 = ax_zoom.boxplot(
        data100, 0, 'gD', showmeans=True, meanline=True,
        patch_artist=True, medianprops=medianprops,
        meanprops=meanprops, widths=0.8)
    for patch, color in zip(bp1['boxes'], colors):
        patch.set_facecolor(color)

    ax_zoom.set_yticks(range(0, 105, 20))
    ax_zoom.set_xticklabels(ip)
    ax1.set_yticks(range(0, 105, 20))
    ax1.set_xlabel('Steps', fontsize="large")
    ax1.set_ylabel('Foraging (%)', fontsize="large")
    ax1.legend(fontsize="small", loc="center right", title='IPs')

    plt.tight_layout()
    maindir = '/tmp/swarm/data/experiments'
    fname = 'ip_paper_efficiency_power_' + str(t)

    fig.savefig(
        maindir + '/' + fname + '.png')
    # pylint: disable = E1101

    plt.close(fig)


def ip_paper_efficiency_power_boxplot(t=5):
    # thresholds = [5, 7, 10, 15]
    ip = [0.5, 0.7, 0.8, 0.85, 0.9, 0.99]
    data100 = [read_data_n_agent_perturbations(
        n=100, iter=12000, threshold=t, time=10000, iprob=i)[:,-1] for i in ip]

    fig = plt.figure(figsize=(8,6), dpi=200)
    ax1 = fig.add_subplot(1, 1, 1)
    colordict = {
        0: 'gold',
        1: 'linen',
        2: 'orchid',
        3: 'peru',
        4: 'olivedrab',
        5: 'indianred',
        6: 'tomato'}
    colorshade = [
        'springgreen', 'lightcoral',
        'khaki', 'lightsalmon', 'deepskyblue']

    # labels = [ "> n/"+str(a) for a in thresholds]
    medianprops = dict(linewidth=2.5, color='firebrick')
    meanprops = dict(linewidth=2.5, color='#ff7f0e')
    bp1 = ax1.boxplot(
        data100, 0, 'gD', showmeans=True, meanline=True,
        patch_artist=True, medianprops=medianprops,
        meanprops=meanprops, widths=0.8)
    for patch, color in zip(bp1['boxes'], colordict.values()):
        patch.set_facecolor(color)

    ax1.legend(zip(bp1['boxes']), ip, fontsize="small", loc="lower right", title='IP')
    # ax1.legend(zip(bp1['boxes']), thresholds, fontsize="small", loc="upper right", title='Storage Threshold')
    # ax1.set_xticks([1.5, 4.5, 7.5, 10.5])
    ax1.set_xticklabels(ip)
    ax1.set_yticks(range(0, 105, 20))
    ax1.set_xlabel('IP', fontsize="large")
    ax1.set_ylabel('Foraging (%)', fontsize="large")
    plt.title('IP')
    plt.tight_layout()
    maindir = '/tmp/swarm/data/experiments'
    fname = 'ipboxplotiprob'+str(t)

    fig.savefig(
        maindir + '/' + fname + '.png')
    # pylint: disable = E1101

    plt.close(fig)


def ip_paper_efficiency_power_lt(t=5):
    ip = [0.5, 0.7, 0.8, 0.85, 0.9, 0.99]
    axeslimitdict = {
        0:[(11500,12002), (75,95)]
        }
    data = [np.mean(np.squeeze(read_data_n_agent_perturbations_all(
        n=100, iter=12000, threshold=t, time=10000, iprob=i,
        addobject='None',no_objects=1, radius=5, idx=[4])), axis=0) for i in ip]

    fig = plt.figure(figsize=(8,6), dpi=200)
    ax1 = fig.add_subplot(1, 1, 1)

    xvalues = np.array(list(range(12002)))
    mask = xvalues % 1000 == 0
    colors = []
    for i in range(len(ip)):
        p = ax1.plot(xvalues[mask], data[i][mask], marker="v", ls='-', label=str(ip[i]))
        colors += [p[0].get_color()]

    ax_zoom = ax1.inset_axes([0.05,0.6,0.8,0.37])
    # ax_zoom = ax1.inset_axes([0.05,0.5,0.47,0.47])
    ax_zoom.patch.set_alpha(0.5)

    medianprops = dict(linewidth=2.5, color='firebrick')
    meanprops = dict(linewidth=2.5, color='#ff7f0e')
    # print(len(data), data[0].shape)
    data = [d[2:].astype(np.int8) for d in data]
    # print(data[0])
    bp1 = ax_zoom.boxplot(
        data, 0, 'gD', showmeans=True, meanline=True,
        patch_artist=True, medianprops=medianprops,
        meanprops=meanprops, widths=0.8)
    for patch, color in zip(bp1['boxes'], colors):
        patch.set_facecolor(color)

    ax_zoom.set_yticks(range(0, 11, 2))
    ax_zoom.set_xticklabels(ip)
    ax1.set_yticks(range(0, 21, 2))
    ax1.set_xlabel('Steps', fontsize="large")
    ax1.set_ylabel('LT Rate (%)', fontsize="large")
    ax1.legend(fontsize="small", loc="upper right", title='IPs')

    plt.tight_layout()
    maindir = '/tmp/swarm/data/experiments'
    fname = 'ip_paper_efficiency_power_lt_' + str(t)

    fig.savefig(
        maindir + '/' + fname + '.png')
    # pylint: disable = E1101

    plt.close(fig)


def ip_paper_efficiency_power_gs(t=5):
    ip = [0.5, 0.7, 0.8, 0.85, 0.9, 0.99]
    axeslimitdict = {
        0:[(11500,12002), (75,95)]
        }
    data = [np.mean(np.squeeze(read_data_n_agent_perturbations_all(
        n=100, iter=12000, threshold=t, time=10000, iprob=i,
        addobject='None',no_objects=1, radius=5, idx=[6])), axis=0) for i in ip]

    fig = plt.figure(figsize=(8,6), dpi=200)
    ax1 = fig.add_subplot(1, 1, 1)

    xvalues = np.array(list(range(12002)))
    mask = xvalues % 1000 == 0
    colors = []
    for i in range(len(ip)):
        p = ax1.plot(xvalues[mask], data[i][mask], marker="v", ls='-', label=str(ip[i]))
        colors += [p[0].get_color()]

    ax_zoom = ax1.inset_axes([0.05,0.6,0.8,0.37])
    ax_zoom.patch.set_alpha(0.5)

    medianprops = dict(linewidth=2.5, color='firebrick')
    meanprops = dict(linewidth=2.5, color='#ff7f0e')
    # print(len(data), data[0].shape)
    data = [d[2:].astype(np.int8) for d in data]
    # print(data[0])
    bp1 = ax_zoom.boxplot(
        data, 0, 'gD', showmeans=True, meanline=True,
        patch_artist=True, medianprops=medianprops,
        meanprops=meanprops, widths=0.8, showfliers=False)
    for patch, color in zip(bp1['boxes'], colors):
        patch.set_facecolor(color)

    ax_zoom.set_yticks(range(0, 60, 10))
    ax_zoom.set_xticklabels(ip)
    ax1.set_yticks(range(0, 111, 10))
    ax1.set_xlabel('Steps', fontsize="large")
    ax1.set_ylabel('GS Rate (%)', fontsize="large")
    ax1.legend(fontsize="small", loc="upper right", title='IPs')

    plt.tight_layout()
    maindir = '/tmp/swarm/data/experiments'
    fname = 'ip_paper_efficiency_power_gs_' + str(t)

    fig.savefig(
        maindir + '/' + fname + '.png')
    # pylint: disable = E1101

    plt.close(fig)


def st_paper_efficiency_power_boxplot_all():
    ips = [0.8, 0.85, 0.9]
    thresholds = [5, 7 , 10, 15]
    datas = {}
    for t in thresholds:
        data100 = [read_data_n_agent_perturbations(
            n=100, iter=12000, threshold=t, time=10000, iprob=ip)[:,-1] for ip in ips]
        datas[t] = data100
    positions = [
        [1, 2, 3], [5, 6, 7], [9, 10, 11], [13, 14, 15]
        ]
    fig = plt.figure(figsize=(6,4), dpi=200)
    ax1 = fig.add_subplot(1, 1, 1)
    colordict = {
        0: 'gold',
        1: 'linen',
        2: 'orchid',
        3: 'peru',
        4: 'olivedrab',
        5: 'indianred',
        6: 'tomato'}
    colorshade = [
        'springgreen', 'lightcoral',
        'khaki', 'lightsalmon', 'deepskyblue']

    # labels = [ "> n/"+str(a) for a in thresholds]
    medianprops = dict(linewidth=2.5, color='firebrick')
    meanprops = dict(linewidth=2.5, color='#ff7f0e')
    for i in range(len(thresholds)):
        # print([d[0].shape for d in datas[i]])
        # print(datas[thresholds[i]], len(datas[thresholds[i]]))
        bp1 = ax1.boxplot(
            datas[thresholds[i]], 0, 'gD', showmeans=True, meanline=True,
            patch_artist=True, medianprops=medianprops,
            meanprops=meanprops, widths=0.8, positions = positions[i])
        for patch, color in zip(bp1['boxes'], colordict.values()):
            patch.set_facecolor(color)

    ax1.legend(zip(bp1['boxes']), ips, fontsize="small", loc="upper right", title='IP')

    ax1.set_xticks([2,6,10,14])
    ax1.set_xticklabels(thresholds)
    ax1.set_yticks(range(0, 105, 20))
    ax1.set_xlabel('Storage Threshold (ST)', fontsize="large")
    ax1.set_ylabel('Foraging (%)', fontsize="large")
    plt.tight_layout()
    maindir = '/tmp/swarm/data/experiments'
    fname = 'stboxplotiproball'

    fig.savefig(
        maindir + '/' + fname + '.png')
    # pylint: disable = E1101

    plt.close(fig)


def compare_learning_notlearning_obstacles(no=1, time=1000):
    dataltstop = np.squeeze(read_data_n_agent_perturbations_all(
        n=100, iter=12000, threshold=7, time=time, iprob=0.85,
        addobject='Obstacles',no_objects=no, radius=10, idx=[2],
        fname='LTStopEvoCoevolutionPPA'))

    data = np.squeeze(read_data_n_agent_perturbations_all(
        n=100, iter=12000, threshold=7, time=time, iprob=0.85,
        addobject='Obstacles',no_objects=no, radius=10, idx=[2]))

    datastoplearn = np.squeeze(read_data_n_agent_perturbations_all(
        n=100, iter=12000, threshold=7, time=time, iprob=0.85,
        addobject='Obstacles',no_objects=no, radius=10, idx=[2],
        fname='LTResumeEvoCoevolutionPPA'))

    fig = plt.figure(figsize=(8,6), dpi=200)
    ax1 = fig.add_subplot(1, 1, 1)
    xvalues = range(12002)

    ax1.plot(
        xvalues, np.median(data, axis=0),
        label="Baseline", color='blue')
    ax1.fill_between(
                xvalues,
                np.quantile(data, q =0.25, axis=0),
                np.quantile(data, q =0.75, axis=0),
                color='DodgerBlue', alpha=0.3)

    ax1.plot(
        xvalues,
        np.median(dataltstop, axis=0) , label="Disabled", color='red')
    ax1.fill_between(
                xvalues,
                np.quantile(dataltstop, q =0.25, axis=0),
                np.quantile(dataltstop, q =0.75, axis=0),
                color='salmon', alpha=0.3)

    ax1.plot(
        xvalues,
        np.median(datastoplearn, axis=0) , label="Enabled After", color='seagreen')
    ax1.fill_between(
                xvalues,
                np.quantile(datastoplearn, q =0.25, axis=0),
                np.quantile(datastoplearn, q =0.75, axis=0),
                color='lime', alpha=0.3)

    ax1.plot([time]*101, range(0,101), 'r--', label='Disabled Triggered')
    ax1.plot([time+1000]*101, range(0,101), 'g--', label='Enable Triggered')
    ax1.legend(fontsize="small", loc="upper left", title='Lateral Transfer')

    ax1.set_xlabel('Evolution Steps')
    ax1.set_ylabel('Foraging (%)')
    ax1.set_yticks(range(0, 105, 20))
    plt.tight_layout()
    maindir = '/tmp/swarm/data/experiments'
    fname = 'ltenabledstop_' + str(no) +'_' + str(time)
    fig.savefig(
        maindir + '/' + fname + '.png')
    # pylint: disable = E1101

    plt.close(fig)


def compare_lt_on_off_no_obst(no=2):
    # nos = [2, 3, 5]
    timings = list(range(1000,9001,2000))
    # dataltstops = dict(zip(timings, [list()] * len(timings)))
    # datas = dict(zip(timings, [list()] * len(timings)))
    dataltstops = []
    datas = []
    datastoplearn = []
    for time in timings:
        dataltstops += [np.squeeze(read_data_n_agent_perturbations_all(
            n=100, iter=12000, threshold=7, time=time, iprob=0.85,
            addobject='Obstacles',no_objects=no, radius=10, idx=[2],
            fname='LTStopEvoCoevolutionPPA'))]

        datas += [np.squeeze(read_data_n_agent_perturbations_all(
            n=100, iter=12000, threshold=7, time=time, iprob=0.85,
            addobject='Obstacles',no_objects=no, radius=10, idx=[2]))]

        datastoplearn += [np.squeeze(read_data_n_agent_perturbations_all(
        n=100, iter=12000, threshold=7, time=time, iprob=0.85,
        addobject='Obstacles',no_objects=no, radius=10, idx=[2],
        fname='LTResumeEvoCoevolutionPPA'))]

    positions = [
        [1, 2, 3], [5, 6, 7], [9, 10, 11], [13, 14, 15], [17, 18, 19]
        ]
    fig = plt.figure(figsize=(8,6), dpi=200)
    ax1 = fig.add_subplot(1, 1, 1)
    colordict = {
        0: 'gold',
        1: 'linen',
        2: 'orchid',
        3: 'peru',
        4: 'olivedrab',
        5: 'indianred',
        6: 'tomato'}
    colorshade = [
        'springgreen', 'lightcoral',
        'khaki', 'lightsalmon', 'deepskyblue']

    medianprops = dict(linewidth=2.5, color='firebrick')
    meanprops = dict(linewidth=2.5, color='#ff7f0e')
    # print(dataltstops[0][:,-1])
    for i in range(len(timings)):

        bp1 = ax1.boxplot(
            [datas[i][:,-1], dataltstops[i][:,-1], datastoplearn[i][:,-1]],
             0, 'gD', showmeans=True, meanline=True,
            patch_artist=True, medianprops=medianprops,
            meanprops=meanprops, widths=0.8, positions = positions[i])
        for patch, color in zip(bp1['boxes'], colordict.values()):
            patch.set_facecolor(color)

    ax1.legend(
        zip(bp1['boxes']), ['Baseline', 'Disabled', 'Enabled After'],
        fontsize="small", loc="center left", title='Lateral Transfer')

    ax1.set_xticks([2,6,10,14, 18])
    ax1.set_xticklabels(timings)
    ax1.set_yticks(range(0, 105, 20))
    ax1.set_xlabel('Triggered Timings', fontsize="large")
    ax1.set_ylabel('Foraging (%)', fontsize="large")
    plt.tight_layout()
    maindir = '/tmp/swarm/data/experiments'
    fname = 'ltonoffboxplotall_' + str(no)

    fig.savefig(
        maindir + '/' + fname + '.png')
    # pylint: disable = E1101

    plt.close(fig)


def ants22_paper_efficiency_power_obstacles(t=7):
    no_obstacles = [1, 3, 5]
    axeslimitdict = {
        0:[(11500,12002), (75,95)]
        }
    # data = [np.mean(np.squeeze(read_data_n_agent_perturbations_all(
    #     n=100, iter=12000, threshold=t, time=1000, iprob=0.85,
    #     addobject='Obstacles',no_objects=i, radius=10, idx=[2])), axis=0) for i in no_obstacles]
    # baseline = np.mean(np.squeeze(read_data_n_agent_perturbations_all(
    #     n=100, iter=12000, threshold=t, time=10000, iprob=0.85,
    #     addobject='None', radius=5, idx=[2])), axis=0)

    data = [read_data_n_agent_perturbations_all(
        n=100, iter=12000, threshold=t, time=1000, iprob=0.85,
        addobject='Obstacles',no_objects=i, radius=10, idx=[2]) for i in no_obstacles]
    baseline = read_data_n_agent_perturbations_all(
        n=100, iter=12000, threshold=t, time=10000, iprob=0.85,
        addobject='None', radius=5, idx=[2])

    meandata = [np.mean(np.squeeze(data[i]), axis=0) for i in range(len(no_obstacles)) ]
    meanbaseline = np.mean(np.squeeze(baseline), axis=0)
    fig = plt.figure(figsize=(8,6), dpi=300)
    ax1 = fig.add_subplot(1, 1, 1)
    # plt.rcParams['text.usetex'] = True
    xvalues = np.array(list(range(12002)))
    mask = xvalues % 1000 == 0
    colors = []
    ax1.plot(
        xvalues[mask], [80]* len(xvalues[mask]), '--',
        label="Threshold ($\Theta$)", alpha=0.5)
    p = ax1.plot(
        xvalues[mask], meanbaseline[mask], marker="v", ls='-', label='0 (Baseline)')
    colors += [p[0].get_color()]
    for i in range(len(no_obstacles)):
        p = ax1.plot(xvalues[mask], meandata[i][mask], marker="v", ls='-', label=str(no_obstacles[i]))
        colors += [p[0].get_color()]

    # Power axis
    ax_power = ax1.inset_axes([0.05,0.68,0.3,0.3])
    ax_power.patch.set_alpha(0.5)

    powerdata100 = [np.squeeze(data[i])[:,-1] for i in range(len(no_obstacles)) ]
    powerbaseline100 = np.squeeze(baseline)[:,-1]
    powerdata100 = [powerbaseline100] + powerdata100
    medianprops = dict(linewidth=2.5, color='firebrick')
    meanprops = dict(linewidth=2.5, color='#ff7f0e')
    bp1 = ax_power.boxplot(
        powerdata100, 0, 'gD', showmeans=True, meanline=True,
        patch_artist=True, medianprops=medianprops,
        meanprops=meanprops, widths=0.6)
    for patch, color in zip(bp1['boxes'], colors):
        patch.set_facecolor(color)

    ax_power.set_yticks(range(70, 105, 5))
    ax_power.set_xticklabels([0] + no_obstacles)
    ax_power.set_title('Power', y=1.0, pad=-140)
    # Efficiency Axis
    ax_eff = ax1.inset_axes([0.68,0.05,0.3,0.3])
    ax_eff.patch.set_alpha(0.5)

    # data100 = [np.squeeze(data[i])[:,-1] for i in range(len(no_obstacles)) ]
    effbaseline100 = np.squeeze(baseline)
    effbaseline100 = [np.squeeze(np.argwhere(effbaseline100[i, :]>=80)) for i in range(effbaseline100.shape[0])]
    effbaseline100 = [np.array([effbaseline100[i][0] if effbaseline100[i].shape[0]>1 else 12000 for i in range(len(effbaseline100)) ])]
    print(effbaseline100, np.min(np.squeeze(np.array(effbaseline100))))
    # min time 5200

    effdata100 = []
    for i in range(len(no_obstacles)):
        effdata = np.squeeze(data[i])
        effdata = [np.squeeze(np.argwhere(effdata[i, :]>=80)) for i in range(effdata.shape[0])]
        effdata = [np.array([effdata[i][0] if effdata[i].shape[0]>1 else 12000 for i in range(len(effdata)) ])]
        effdata100 += effdata

    effdata100 = effbaseline100 + effdata100

    bp1 = ax_eff.boxplot(
        effdata100, 0, 'gD', showmeans=True, meanline=True,
        patch_artist=True, medianprops=medianprops,
        meanprops=meanprops, widths=0.6, vert=False)
    for patch, color in zip(bp1['boxes'], colors):
        patch.set_facecolor(color)

    ax_eff.set_xticks(range(5000, 12000, 2000))
    ax_eff.set_yticklabels([0] + no_obstacles)
    ax_eff.set_title('Efficiency')

    ax1.set_yticks(range(0, 105, 20))
    ax1.set_xlabel('Steps', fontsize="large")
    ax1.set_ylabel('Foraging (%)', fontsize="large")
    ax1.legend(fontsize="small", loc="lower left", title='No. of Obstacles')

    plt.tight_layout()
    maindir = '/tmp/swarm/data/experiments'
    fname = 'efficiency_power_ablation_obstacles'

    fig.savefig(
        maindir + '/' + fname + '.png')
    # pylint: disable = E1101

    plt.close(fig)


def main():
    # plot_evolution_algo_performance_boxplot()
    # plot_evolution_algo_performance()
    # withWithoutLt()
    # storage_threshold_new()
    # exploration_parameter()
    # generationstep_parameter()
    # plot_none_obstacles_trap()
    # plot_obstacles_time()
    # plot_lt_rate()
    # [0.1, 0.2, 0.3, 0.4,
    # for ip in [0.5, 0.6, 0.7, 0.8, 0.9, 0.99]:
    # for ip in [0.85]:
    #     # storage_threshold_iprob(ip=ip)
    #     # storage_threshold_iprob_lt(ip=ip)
    #     # storage_threshold_iprob_gs(ip=ip)
    #     # storage_threshold_all_100(ip=ip)
    #     # plot_lt_foraging_gentic(ip=ip)
    #     st_paper_efficiency_power_boxplot(ip=ip)
    # obstacle_introduced()
    # trap_introduced()
    # trap_introduced_deadagent()
    # obstacle_introduced_compare()
    # for t in range(1000,11001,1000):
    #    plot_lt_foraging_gentic(time=t)

    # plot_foraging_baseline()
    # ip_paper_efficiency_power_boxplot_unified()

    # basedata = np.squeeze(read_data_n_agent_perturbations_all(
    #     n=100, iter=12000, threshold=7, time=10000, iprob=0.85,
    #     no_objects=1, radius=5, idx=[2,3]))

    # # # plot_foraging_deadagent_curve(basedata)
    # for t in range(1000,4001,1000):
    #     plot_foraging_baseline_obstacles(basedata, time=t)
    # #     # plot_foraging_baseline_trap(basedata, time=t)
    # #     # plot_foraging_baseline_trap_deadagent(basedata, time=t)

    # # plot_no_obstacles_performance(time=2000)
    # for t in [5]:
    #     # ip_paper_efficiency_power_lt(t)
    #     # ip_paper_efficiency_power_gs(t)
    #     ip_paper_efficiency_power(t)
    # #     # ip_paper_efficiency_power_boxplot(t)
    # st_paper_efficiency_power_boxplot_all()
    # for i in range(1000,11001,1000):
    #     compare_learning_notlearning_obstacles(no=2, time=i)
    # for i in [2, 4]:
    #     compare_lt_on_off_no_obst(no=i)

    # ants22_paper_efficiency_power_obstacles(t=7)
    # ip_paper_efficiency_power(t=7)
    # plot_shift_foraging_lt_on_off()
    st_paper_efficiency_power_boxplot_all()


if __name__ == '__main__':
    main()