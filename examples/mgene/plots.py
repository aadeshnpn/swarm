import os
from turtle import position
import numpy as np
import scipy.stats as stats
import matplotlib

# If there is $DISPLAY, display the plot
if os.name == 'posix' and "DISPLAY" not in os.environ:
    matplotlib.use('Agg')

import matplotlib.pyplot as plt     # noqa: E402
import glob
import pathlib


# def read_data_sample_ratio_complex_geese(ratio=0.1, end=True):
#     maindir = '/tmp/samplingcomparision/complex-geese/' # + str(ratio)
#     data = []
#     # print(maindir)
#     folders = pathlib.Path(maindir).glob("*CombinedModelPPA")
#     for f in folders:
#         try:
#             # print(f, ratio, f.joinpath(str(ratio)))
#             nested_folder = pathlib.Path(f.joinpath(str(ratio))).glob("*CoevoSimulation")
#             for f1 in nested_folder:
#                 flist = [p for p in pathlib.Path(f1).iterdir() if p.is_file() and p.match('simulation.csv')]
#                 # print(flist)
#                 _, _, d = np.genfromtxt(flist[0], autostrip=True, unpack=True, delimiter='|')
#                 # print(d.shape)
#                 if end:
#                     data.append(d[-1])
#                 else:
#                     data.append(d)
#         except:
#             pass
#     data = np.array(data)
#     # print(ratio, data.shape, data)
#     return data


def read_data_sample_ratio_complex_geese(ratio=0.1):
    maindir = '/home/aadeshnpn/Desktop/plots_ants22j/ratios/complex-geese/' + str(ratio)
    data = []
    # print(maindir)
    folders = pathlib.Path(maindir).glob("*CoevoSimulation")
    for f in folders:
        try:
            # print(f, ratio, f.joinpath(str(ratio)))
            flist = [p for p in pathlib.Path(f).iterdir() if p.is_file() and p.match('simulation.csv')]
            # print(flist)
            _, _, d = np.genfromtxt(flist[0], autostrip=True, unpack=True, delimiter='|')
            # print(d.shape)
            data.append(d[-1])
        except:
            pass
    data = np.array(data)
    # print(ratio, data.shape)
    return data


def read_data_sample_comparision(
        suffix="geese-bt", simname="*EvoSForge_3",
        folder="learning_comparision"):
    maindir = '/home/aadeshnpn/Desktop/plots_ants22j/'+ folder + '/' + suffix
    data = []
    # print(maindir, simname)
    folders = pathlib.Path(maindir).glob(simname)
    for f in folders:
        try:
            # print(f)
            flist = [p for p in pathlib.Path(f).iterdir() if p.is_file() and p.match('simulation.csv')]
            # print(flist)
            datas = np.genfromtxt(flist[0], autostrip=True, unpack=True, delimiter='|')
            # print(datas.shape)
            if len(datas[2]) <= 12000:
                temp_data = np.array([0]* 12001)
                temp_data[: len(datas[2])] = datas[2]
                temp_data[len(datas[2]):] = np.array([datas[2][-1]] * (12001 - len(datas[2])))
                # print(temp_data.shape, temp_data[11990:])
                temp_data[0] = 0
                data.append(temp_data)
            else:
                data.append(datas[2])
        except:
            pass
    data = np.array(data)
    print(suffix, data.shape)
    return data


def read_data_sample_runtime(
        suffix="geese-bt", simname="*EvoSForge_3",
        folder="learning_comparision"):
    maindir = '/home/aadeshnpn/Desktop/plots_ants22j/'+ folder + '/' + suffix
    data = []
    # print(maindir, simname)
    folders = pathlib.Path(maindir).glob(simname)
    for f in folders:
        try:
            # print(f)
            flist = [p for p in pathlib.Path(f).iterdir() if p.is_file() and p.match('simulation.csv')]
            # print(flist)
            datas = np.genfromtxt(flist[0], autostrip=True, unpack=True, delimiter='|')
            time = datas[3][-1] - datas[3][1]
            data.append(time/60.0)
        except:
            pass
    data = np.array(data)
    print(suffix, data.shape)
    return data


def read_data_restrictive_grammar():
    maindir = '/home/aadeshnpn/Desktop/plots_ants22j/restrictivegrammar/'
    data = []
    # print(maindir)
    folders = pathlib.Path(maindir).glob("*EvoCoevolutionPPA")
    for f in folders:
        try:
            flist = [p for p in pathlib.Path(f).iterdir() if p.is_file() and p.match('simulation.csv')]
            # print(flist)
            datas = np.genfromtxt(flist[0], autostrip=True, unpack=True, delimiter='|')
            # print(datas.shape)
            temp_data = np.array([0]* 12001)
            temp_data[: len(datas[2])] = datas[2]
            temp_data[len(datas[2]):] = np.array([datas[2][-1]] * (12001 - len(datas[2])))
            # print(temp_data.shape, temp_data[11990:])
            temp_data[0] = 0
            data.append(temp_data)
        except:
            pass
    data = np.array(data)
    return data


def read_data_sample_ratio_geesebt(ratio=0.1, end=True):
    maindir = '/home/aadeshnpn/Desktop/plots_ants22j/ratios/geese-bt/'
    folders = pathlib.Path(maindir).glob("*[0-9]*-" + str(ratio))

    flist = []
    data = []
    for f in folders:
        try:
            flist = [p for p in pathlib.Path(f).iterdir() if p.is_file() and p.match('simulation.csv')]
            _, _, d = np.genfromtxt(flist[0], autostrip=True, unpack=True, delimiter='|')
            if end:
                data.append(d[-1])
            else:
                data.append(d)
        except:
            pass
    data = np.array(data)
    return data


def read_data_sample_ratio_betrgeese(ratio=0.1, end=True):
    maindir = '/home/aadeshnpn/Desktop/plots_ants22j/ratios/betr-geese/'
    folders = pathlib.Path(maindir).glob("*_" + str(ratio) + "_ValidateSForgeNewPPA1")
    flist = []
    data = []
    for f in folders:
        try:
            flist = [p for p in pathlib.Path(f).iterdir() if p.is_file() and p.match('simulation.csv')]
            _, _, d = np.genfromtxt(flist[0], autostrip=True, unpack=True, delimiter='|')
            if end:
                data.append(d[-1])
            else:
                data.append(d)
        except:
            pass
    data = np.array(data)
    return data


def compare_all_geese_efficiency():

    suffixs = ["geese-bt", "betr-geese", "complex-geese"]
    labels = ["GEESE-BT", "BeTr-GEESE", "Multi-GEESE"]
    simnames = ["*EvoSForge_3", "*EvoCoevolutionPPA", "*CombinedModelPPA"]

    color = [
        'indianred', 'forestgreen', 'gold',
        'tomato', 'royalblue']
    colorshade = [
        'coral', 'springgreen', 'yellow',
        'lightsalmon', 'deepskyblue']

    # plt.style.use('fivethirtyeight')
    fig = plt.figure(figsize=(10, 6), dpi=300)
    ax1 = fig.add_subplot(1, 1, 1)

    for j in range(len(suffixs)):
        data = read_data_sample_comparision(suffixs[j], simnames[j])

        medianf = np.quantile(data, 0.5, axis=0)
        q1f = np.quantile(data, 0.25, axis=0)
        q3f = np.quantile(data, 0.75, axis=0)


        xvalues = range(data.shape[1])
        ax1.plot(
            xvalues, medianf, # color=color[j],
            linewidth=1.0, label=labels[j])
        ax1.fill_between(
            xvalues, q3f, q1f,
            alpha=0.3)

    ax1.set_xlabel('Steps')
    ax1.set_ylabel('Foraging (%)')
    ax1.set_yticks(range(0,110,20))
    # ax1.set_xticks(
    #     np.linspace(0, data.shape[-1], 5))
    plt.legend(loc="upper left")
    plt.tight_layout()

    # fig.savefig(
    #     '/tmp/goal/data/experiments/' + pname + '.pdf')
    maindir = '/tmp/swarm/data/experiments/'
    # nadir = os.path.join(maindir, str(n), agent)
    fig.savefig(
        maindir + 'all_geese_comparasion.png')
    plt.close(fig)


def power_efficiency_slides():
    suffixs = ["geese-bt", "betr-geese", "complex-geese"]
    simnames = ["*EvoSForge_3", "*EvoCoevolutionPPA", "*CombinedModelPPA"]

    color = [
        'indianred', 'forestgreen', 'gold',
        'tomato', 'royalblue']
    colorshade = [
        'coral', 'springgreen', 'yellow',
        'lightsalmon', 'deepskyblue']

    # plt.style.use('fivethirtyeight')
    fig = plt.figure(figsize=(10, 6), dpi=300)
    ax1 = fig.add_subplot(1, 1, 1)

    data = read_data_sample_comparision(suffixs[1], simnames[1])
    maxv_idx = np.argmax(np.max(data[:, 1:], axis=1))
    # print(maxvalues.shape, maxvalues)
    sample = data[maxv_idx]
    t_theta = np.where(sample>80)
    print(t_theta)
    xvalues = range(data.shape[1])
    ax1.plot(
        xvalues, sample, # color=color[j],
        linewidth=1.0, label='BeTr-GEESE')
    ax1.plot(
        xvalues, [80]*len(xvalues), color='green', label=r'$\theta$')

    ax1.plot(
        [t_theta[0][0]]*100, range(100), color='red', label=r'$t_{\theta}$')
    ax1.set_xlabel('Steps')
    ax1.set_ylabel('Foraging (%)')
    ax1.set_yticks(range(0,110,20))
    # ax1.set_xticks(
    #     np.linspace(0, data.shape[-1], 5))
    plt.legend(loc="upper left")
    plt.tight_layout()

    # fig.savefig(
    #     '/tmp/goal/data/experiments/' + pname + '.pdf')
    maindir = '/tmp/swarm/data/experiments/'
    # nadir = os.path.join(maindir, str(n), agent)
    fig.savefig(
        maindir + 'power_efficiency_slide.png')
    plt.close(fig)
    pass


def compare_all_fixed_behaviors_efficiency():

    suffixs = ["geese-bt", "betr-geese", "complex-geese"]
    simnames = ["*ValidateSForge","*_0.5_ValidateSForgeNewPPA1", "*CoevoSimulation"]

    color = [
        'indianred', 'forestgreen', 'gold',
        'tomato', 'royalblue']
    colorshade = [
        'coral', 'springgreen', 'yellow',
        'lightsalmon', 'deepskyblue']

    # plt.style.use('fivethirtyeight')
    fig = plt.figure(figsize=(10, 4), dpi=300)
    ax1 = fig.add_subplot(1, 2, 1)
    labels = ['Top Agents (GEESE-BT)', 'Parallel (BeTr-GEESE)', 'Top Agents (Multi-GEESE)']

    for j in range(len(suffixs)):
        data = read_data_sample_comparision(
            suffixs[j], simnames[j], folder="fixed_behavior_efficiency")

        medianf = np.quantile(data, 0.5, axis=0)
        q1f = np.quantile(data, 0.25, axis=0)
        q3f = np.quantile(data, 0.75, axis=0)


        xvalues = range(data.shape[1])
        ax1.plot(
            xvalues, medianf, # color=color[j],
            linewidth=1.0, label=labels[j])
        ax1.fill_between(
            xvalues, q3f, q1f,
            alpha=0.3)

    ax1.set_xlabel('Steps')
    ax1.set_ylabel('Foraging (%)')
    ax1.set_yticks(range(0,110,20))
    # ax1.set_xticks(
    #     np.linspace(0, data.shape[-1], 5))
    ax1.legend(loc="upper left")

    suffixs = ["geese-bt", "betr-geese", "complex-geese"]
    simnames = ["*ValidateSForge","*CoevoSimulation", "*CoevoSimulation"]
    datas = []

    for j in range(3):
        data = read_data_sample_runtime(
            suffixs[j], simnames[j], folder="computation_time")
        datas.append(data)

    ax2 = fig.add_subplot(1, 2, 2)
    colordict = {
        0: 'forestgreen',
        1: 'gold',
        2: 'royalblue',
        3: 'orchid',
        4: 'olivedrab',
        5: 'peru',
        6: 'linen',
        7: 'indianred',
        8: 'tomato'}

    medianprops = dict(linewidth=2.5, color='firebrick')
    meanprops = dict(linewidth=2.5, color='#ff7f0e')
    bp1 = ax2.boxplot(
        datas, 0, 'gD', showmeans=True, meanline=True,
        patch_artist=True, medianprops=medianprops,
        meanprops=meanprops, widths=0.3)
    for patch, color in zip(bp1['boxes'], colordict.values()):
        patch.set_facecolor(color)

    # ax2.legend(
    #     zip(bp1['boxes']), labels,
    #     fontsize="small", loc="center left")
    ax2.set_xticklabels(['GEESE-BT', 'BeTr-GEESE', 'Multi-GEESE'])
    ax2.set_ylabel('Run Time (Minutes)',  fontsize="large")

    # ax1.indicate_inset_zoom(ax_zoom, edgecolor="black", label="Zoomed", alpha=0.3)
    plt.tight_layout()
    maindir = '/tmp/swarm/data/experiments/'
    fig.savefig(
        maindir + 'all_geese_fixed_behaviors_comparasion.png')

    plt.close(fig)


def compare_run_time():
    suffixs = ["geese-bt", "betr-geese", "complex-geese"]
    simnames = ["*ValidateSForge","*CoevoSimulation", "*CoevoSimulation"]
    datas = []
    for j in range(3):
        data = read_data_sample_runtime(
            suffixs[j], simnames[j], folder="computation_time")
        datas.append(data)

    fig = plt.figure(figsize=(8,6), dpi=300)
    ax1 = fig.add_subplot(1, 1, 1)

    colordict = {
        0: 'forestgreen',
        1: 'gold',
        2: 'royalblue',
        3: 'orchid',
        4: 'olivedrab',
        5: 'peru',
        6: 'linen',
        7: 'indianred',
        8: 'tomato'}

    colorshade = [
        'springgreen', 'lightcoral',
        'khaki', 'lightsalmon', 'deepskyblue']

    medianprops = dict(linewidth=2.5, color='firebrick')
    meanprops = dict(linewidth=2.5, color='#ff7f0e')
    bp1 = ax1.boxplot(
        datas, 0, 'gD', showmeans=True, meanline=True,
        patch_artist=True, medianprops=medianprops,
        meanprops=meanprops, widths=0.8)
    for patch, color in zip(bp1['boxes'], colordict.values()):
        patch.set_facecolor(color)

    # plt.xlim(0, len(mean))
    labels = ['Top Agents (GEESE-BT)', 'Parallel (BeTr-GEESE)', 'Top Agents (Complex-GEESE)']
    ax1.legend(
        zip(bp1['boxes']), labels,
        fontsize="small", loc="upper right", title='Sampling Algorithm')
    # ax1.set_xticks([2, 6, 10, 14, 18])
    ax1.set_xticklabels(labels)
    # ax1.set_yticks(range(0, 105, 20))
    # ax1.set_xlabel('Sampling Size', fontsize="large")
    ax1.set_ylabel('Run Time (Minutes)',  fontsize="large")
    # ax1.set_title('Behavior Sampling',  fontsize="large")

    plt.tight_layout()

    maindir = '/tmp/swarm/data/experiments/'
    fname = 'run_time'

    fig.savefig(
        maindir + '/' + fname + '.png')

    plt.close(fig)


def compare_sampling_differences():
    # plt.style.use('fivethirtyeight')
    sampling_size = [0.1, 0.3, 0.5, 0.7, 0.9]
    datasnew = [read_data_sample_ratio_betrgeese(s) for s in sampling_size]
    datasold = [read_data_sample_ratio_geesebt(s) for s in sampling_size]
    datascom = [read_data_sample_ratio_complex_geese(s) for s in sampling_size]
    # print(datasnew, datasold)
    fig = plt.figure(figsize=(8,6), dpi=300)
    ax1 = fig.add_subplot(1, 1, 1)

    colordict = {
        0: 'forestgreen',
        1: 'gold',
        2: 'royalblue',
        3: 'orchid',
        4: 'olivedrab',
        5: 'peru',
        6: 'linen',
        7: 'indianred',
        8: 'tomato'}

    colorshade = [
        'springgreen', 'lightcoral',
        'khaki', 'lightsalmon', 'deepskyblue']
    labels = sampling_size
    positions = [
        [1, 2, 3], [5, 6, 7], [9, 10, 11], [13, 14, 15], [17, 18, 19] # , [16, 17]
        ]
    datas = [
        [datasold[0], datasnew[0], datascom[0] ],
        [datasold[1], datasnew[1], datascom[1] ],
        [datasold[2], datasnew[2], datascom[2] ],
        [datasold[3], datasnew[3], datascom[3] ],
        [datasold[4], datasnew[4], datascom[4] ],
        # [datasold[5],datasnew[5]],
    ]

    medianprops = dict(linewidth=2.5, color='firebrick')
    meanprops = dict(linewidth=2.5, color='#ff7f0e')
    for i in range(len(positions)):
        bp1 = ax1.boxplot(
            datas[i], 0, 'gD', showmeans=True, meanline=True,
            patch_artist=True, medianprops=medianprops,
            meanprops=meanprops, positions=positions[i], widths=0.8)
        for patch, color in zip(bp1['boxes'], colordict.values()):
            patch.set_facecolor(color)

    # plt.xlim(0, len(mean))
    ax1.legend(
        zip(bp1['boxes']), ['Top Agents (GEESE-BT)', 'Parallel (BeTr-GEESE)', 'Top Agents (Multi-GEESE)'],
        fontsize="small", loc="upper right", title='Sampling Algorithm')
    ax1.set_xticks([2, 6, 10, 14, 18])
    ax1.set_xticklabels(labels)
    ax1.set_yticks(range(0, 105, 20))
    ax1.set_xlabel('Sampling Size', fontsize="large")
    ax1.set_ylabel('Maintenance (%)',  fontsize="large")
    # ax1.set_title('Behavior Sampling',  fontsize="large")

    plt.tight_layout()

    maindir = '/tmp/swarm/data/experiments/'
    fname = 'behavior_sampling'

    fig.savefig(
        maindir + '/' + fname + '.png')

    plt.close(fig)


def compare_sampling_differences_plot():
    # plt.style.use('fivethirtyeight')
    sampling_size = [0.1, 0.3, 0.5, 0.7, 0.9]

    datasold_forge = [read_data_sample_ratio_geesebt(s) for s in sampling_size]
    datasnew_forge = [read_data_sample_ratio_betrgeese(s) for s in sampling_size]
    datacomplex_forge = [read_data_sample_ratio_complex_geese(s) for s in sampling_size]
    fig = plt.figure(figsize=(8,6), dpi=300)
    ax1 = fig.add_subplot(1, 1, 1)

    sampling_size = [10, 30, 50, 70, 90]
    colordict = {
        0: 'forestgreen',
        1: 'gold',
        2: 'royalblue',
        3: 'orchid',
        4: 'olivedrab',
        5: 'peru',
        6: 'linen',
        7: 'indianred',
        8: 'tomato'}

    colorshade = [
        'springgreen', 'lightcoral',
        'khaki', 'lightsalmon', 'deepskyblue']
    labels = sampling_size

    positions = [1.5, 4.5, 7.5, 10.5, 13.5]
    # datas = [
    #     [datasold[0],datasnew[0]],
    #     [datasold[1],datasnew[1]],
    #     [datasold[2],datasnew[2]],
    #     [datasold[3],datasnew[3]],
    #     [datasold[4],datasnew[4]],
    # ]

    datasforge = [
        [datasold_forge[0],datasnew_forge[0], datacomplex_forge[0]],
        [datasold_forge[1],datasnew_forge[1], datacomplex_forge[1]],
        [datasold_forge[2],datasnew_forge[2], datacomplex_forge[2]],
        [datasold_forge[3],datasnew_forge[3], datacomplex_forge[3]],
        [datasold_forge[4],datasnew_forge[4], datacomplex_forge[4]],
    ]

    colors = ['hotpink', 'mediumblue', 'olivedrab']
    plabels = ['Top Agents (GEESE-BT)', 'Parallel (BeTr-GEESE)', 'Top Agents (Multi-GEESE)']
    # lss = ['--', '-']
    markers=['o', '^']
    # for i in range(2):
        # p = ax1.plot(
        #     [x for x in positions],
        #     [np.mean(d[i]) for d in datas], marker=markers[0],
        #     ls='-', label=plabels[i], color=colors[i], markersize=10, linewidth=3)
        # colors += [p[0].get_color()]

    for i in range(3):
        p = ax1.plot(
            [x for x in positions],
            [np.mean(d[i]) for d in datasforge], marker=markers[1],
            ls='-', label=plabels[i], color = colors[i], markersize=10, linewidth=3)
        # colors += [p[0].get_color()]

    f = lambda m,c: plt.plot([],[],marker=m, color=c, ls="none")[0]

    handles1 = [f('_', colors[i]) for i in range(3)]
    handles2 = [f(markers[1], 'k') for i in range(2)]

    # legendlabels = plabels + ["Foraging", "Maintenance"]
    # plt.xlim(0, len(mean))
    legend1 = ax1.legend(handles1, plabels,
        fontsize="large", loc="upper right", title='Sampling Algorithm', markerscale=6)

    legend2 = ax1.legend(handles2, ["Foraging"],
        fontsize="large", loc="upper left", title='Task')

    plt.setp(legend1.get_title(),fontsize='large')
    plt.setp(legend2.get_title(),fontsize='large')
    ax1.add_artist(legend1)
    ax1.add_artist(legend2)
    ax1.set_xticks([1.5, 4.5, 7.5, 10.5, 13.5])
    ax1.set_xticklabels(labels, fontsize='large')
    ax1.set_yticks(range(0, 105, 20))
    ax1.set_yticklabels(range(0, 105, 20), fontsize='large')
    ax1.set_xlabel('Sampling Size (%)', fontsize="large")
    ax1.set_ylabel('Performance (%)',  fontsize="large")
    # ax1.set_title('Behavior Sampling',  fontsize="large")
    # plt.rcParams['legend.title_fontsize'] = 'large'
    plt.tight_layout()

    maindir = '/tmp/swarm/data/experiments'
    fname = 'behavior_samplingnest_agg'

    fig.savefig(
        maindir + '/' + fname + '.png')

    plt.close(fig)


def plot_restictive_grammar():
    # plt.style.use('fivethirtyeight')
    fig = plt.figure(figsize=(10, 6), dpi=300)
    ax1 = fig.add_subplot(1, 1, 1)

    data = read_data_restrictive_grammar()
    print(data)
    medianf = np.quantile(data, 0.5, axis=0)
    q1f = np.quantile(data, 0.25, axis=0)
    q3f = np.quantile(data, 0.75, axis=0)

    xvalues = range(data.shape[1])
    ax1.plot(
        xvalues, medianf, # color=color[j],
        linewidth=1.0, label='Complex-GEESE')
    ax1.fill_between(
        xvalues, q3f, q1f,
        alpha=0.3)

    ax1.set_xlabel('Steps')
    ax1.set_ylabel('Foraging (%)')
    ax1.set_yticks(range(0,110,20))
    # ax1.set_xticks(
    #     np.linspace(0, data.shape[-1], 5))
    plt.legend(loc="upper left")
    plt.tight_layout()

    # fig.savefig(
    #     '/tmp/goal/data/experiments/' + pname + '.pdf')
    maindir = '/tmp/swarm/data/experiments/'
    # nadir = os.path.join(maindir, str(n), agent)
    fig.savefig(
        maindir + 'complex_geese.png')
    plt.close(fig)


def learning_efficience_sampling_combined():
    suffixs = ["geese-bt", "betr-geese", "complex-geese"]
    labels = ["GEESE-BT", "BeTr-GEESE", "Multi-GEESE"]
    simnames = ["*EvoSForge_3", "*EvoCoevolutionPPA", "*CombinedModelPPA"]
    fig = plt.figure(figsize=(12, 6), dpi=300)
    ax2 = fig.add_subplot(1, 2, 1)

    for j in range(len(suffixs)):
        data = read_data_sample_comparision(suffixs[j], simnames[j])

        medianf = np.quantile(data, 0.5, axis=0)
        q1f = np.quantile(data, 0.25, axis=0)
        q3f = np.quantile(data, 0.75, axis=0)


        xvalues = range(data.shape[1])
        ax2.plot(
            xvalues, medianf, # color=color[j],
            linewidth=1.0, label=labels[j])
        ax2.fill_between(
            xvalues, q3f, q1f,
            alpha=0.3)

    ax2.set_xlabel('Steps', fontsize='large')
    ax2.set_ylabel('Foraging (%)', fontsize='large')
    ax2.set_yticks(range(0,110,20))
    ax2.legend(loc="upper left", fontsize='large')

    sampling_size = [0.1, 0.3, 0.5, 0.7, 0.9]
    datasold_forge = [read_data_sample_ratio_geesebt(s) for s in sampling_size]
    datasnew_forge = [read_data_sample_ratio_betrgeese(s) for s in sampling_size]
    datacomplex_forge = [read_data_sample_ratio_complex_geese(s) for s in sampling_size]

    ax1 = fig.add_subplot(1, 2, 2)

    sampling_size = [10, 30, 50, 70, 90]
    labels = sampling_size

    positions = [1.5, 4.5, 7.5, 10.5, 13.5]

    datasforge = [
        [datasold_forge[0],datasnew_forge[0], datacomplex_forge[0]],
        [datasold_forge[1],datasnew_forge[1], datacomplex_forge[1]],
        [datasold_forge[2],datasnew_forge[2], datacomplex_forge[2]],
        [datasold_forge[3],datasnew_forge[3], datacomplex_forge[3]],
        [datasold_forge[4],datasnew_forge[4], datacomplex_forge[4]],
    ]

    colors = ['hotpink', 'mediumblue', 'olivedrab']
    plabels = ['Top Agents (GEESE-BT)', 'Parallel (BeTr-GEESE)', 'Top Agents (Multi-GEESE)']
    markers=['o', '^']

    for i in range(3):
        p = ax1.plot(
            [x for x in positions],
            [np.mean(d[i]) for d in datasforge], marker=markers[1],
            ls='-', label=plabels[i], color = colors[i], markersize=10, linewidth=3)

    f = lambda m,c: plt.plot([],[],marker=m, color=c, ls="none")[0]

    handles1 = [f('_', colors[i]) for i in range(3)]
    handles2 = [f(markers[1], 'k') for i in range(2)]

    legend1 = ax1.legend(handles1, plabels,
        fontsize="large", loc="upper right", title='Sampling Algorithm', markerscale=6)

    # legend2 = ax1.legend(handles2, ["Foraging"],
    #     fontsize="large", loc="upper left", title='Task')

    plt.setp(legend1.get_title(),fontsize='large')
    # plt.setp(legend2.get_title(),fontsize='large')
    ax1.add_artist(legend1)
    # ax1.add_artist(legend2)
    ax1.set_xticks([1.5, 4.5, 7.5, 10.5, 13.5])
    ax1.set_xticklabels(labels, fontsize='large')
    ax1.set_yticks(range(0, 105, 20))
    ax1.set_yticklabels(range(0, 105, 20), fontsize='large')
    ax1.set_xlabel('Sampling Size (%)', fontsize="large")
    ax1.set_ylabel('Performance (%)',  fontsize="large")

    plt.tight_layout()

    # fig.savefig(
    #     '/tmp/goal/data/experiments/' + pname + '.pdf')
    maindir = '/tmp/swarm/data/experiments/'
    # nadir = os.path.join(maindir, str(n), agent)
    fig.savefig(
        maindir + 'learning_efficiency_sampling.png')
    plt.close(fig)


def debug_learning_efficience():
    suffixs = ["complex-geese"]
    labels = ["Multi-GEESE"]
    simnames = ["*CombinedModelPPA"]
    fig = plt.figure(figsize=(6, 3), dpi=100)
    ax2 = fig.add_subplot(1, 1, 1)

    for j in range(len(suffixs)):
        data = read_data_sample_comparision(suffixs[j], simnames[j])

        medianf = np.quantile(data, 0.5, axis=0)
        q1f = np.quantile(data, 0.25, axis=0)
        q3f = np.quantile(data, 0.75, axis=0)


        xvalues = range(data.shape[1])
        ax2.plot(
            xvalues, medianf, # color=color[j],
            linewidth=1.0, label=labels[j])
        ax2.fill_between(
            xvalues, q3f, q1f,
            alpha=0.3)

    ax2.set_xlabel('Steps', fontsize='large')
    ax2.set_ylabel('Foraging (%)', fontsize='large')
    ax2.set_yticks(range(0,110,20))
    ax2.legend(loc="upper left", fontsize='large')
    plt.tight_layout()

    # fig.savefig(
    #     '/tmp/goal/data/experiments/' + pname + '.pdf')
    maindir = '/tmp/swarm/data/experiments/'
    # nadir = os.path.join(maindir, str(n), agent)
    fig.savefig(
        maindir + 'debug_learning_efficiency.png')
    plt.show()
    # plt.close(fig)


def main():
    # fixed_behaviors_sampling_ratio()
    # compare_all_geese_efficiency()
    # compare_sampling_differences_plot()
    # compare_sampling_differences()
    # read_data_sample_ratio_complex_geese1(0.1)
    # plot_restictive_grammar()
    # compare_all_fixed_behaviors_efficiency()
    # compare_run_time()
    # power_efficiency_slides()
    # learning_efficience_sampling_combined()
    debug_learning_efficience()


if __name__ == '__main__':
    main()