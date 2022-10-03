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


def read_data_sample_ratio_complex_geese(ratio=0.1):
    maindir = '/tmp/samplingcomparision/complex-geese/' # + str(ratio)
    data = []
    # print(maindir)
    folders = pathlib.Path(maindir).glob("*CombinedModelPPA")
    for f in folders:
        try:
            # print(f, ratio, f.joinpath(str(ratio)))
            nested_folder = pathlib.Path(f.joinpath(str(ratio))).glob("*CoevoSimulation")
            for f1 in nested_folder:
                flist = [p for p in pathlib.Path(f1).iterdir() if p.is_file() and p.match('simulation.csv')]
                # print(flist)
                _, _, d = np.genfromtxt(flist[0], autostrip=True, unpack=True, delimiter='|')
                # print(d.shape)
                data.append(d[-1])
        except:
            pass
    data = np.array(data)
    # print(ratio, data.shape, data)
    return data


def read_data_sample_comparision(suffix="geese-bt", simname="*EvoSForge_3"):
    maindir = '/home/aadeshnpn/Desktop/plots_ants22j/learning_comparision/'+ suffix
    data = []
    # print(maindir)
    folders = pathlib.Path(maindir).glob(simname)
    for f in folders:
        try:
            flist = [p for p in pathlib.Path(f).iterdir() if p.is_file() and p.match('simulation.csv')]
            # print(flist)
            datas = np.genfromtxt(flist[0], autostrip=True, unpack=True, delimiter='|')
            # print(datas.shape)
            data.append(datas[2])
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


def read_data_sample_ratio_geesebt(ratio=0.1):
    maindir = '/tmp/samplingcomparision/bsample/'
    folders = pathlib.Path(maindir).glob("*_" + str(ratio) + "_ValidateSForgeNewPPA1")
    flist = []
    data = []
    for f in folders:
        try:
            flist = [p for p in pathlib.Path(f).iterdir() if p.is_file() and p.match('simulation.csv')]
            _, _, d = np.genfromtxt(flist[0], autostrip=True, unpack=True, delimiter='|')
            data.append(d[-1])
        except:
            pass
    data = np.array(data)
    return data


def read_data_sample_ratio_betrgeese(ratio=0.1):
    maindir = '/tmp/samplingcomparision/ratioijcai/'
    folders = pathlib.Path(maindir).glob("*[0-9]*-" + str(ratio))
    flist = []
    data = []
    for f in folders:
        try:
            flist = [p for p in pathlib.Path(f).iterdir() if p.is_file() and p.match('simulation.csv')]
            _, _, d = np.genfromtxt(flist[0], autostrip=True, unpack=True, delimiter='|')
            data.append(d[-1])
        except:
            pass
    data = np.array(data)
    return data


def fixed_behaviors_sampling_ratio():
    ratios = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99]
    data = [read_data_sample_ratio(r) for r in ratios]

    fig = plt.figure(figsize=(8,6), dpi=200)
    ax1 = fig.add_subplot(1, 1, 1)
    medianprops = dict(linewidth=2.5, color='firebrick')
    meanprops = dict(linewidth=2.5, color='#ff7f0e')

    colordict = {
        0: 'grey',
        1: 'whitesmoke',
        2: 'rosybrown',
        3: 'darkred',
        4: 'darkorange',
        5: 'olive',
        6: 'yellow',
        7: 'lawngreen',
        8: 'aqua',
        9: 'indigo',
        }

    bp1 = ax1.boxplot(
        data, 0, 'gD', showmeans=True, meanline=True,
        patch_artist=True, medianprops=medianprops,
        meanprops=meanprops, widths=0.8)

    for patch, color in zip(bp1['boxes'], colordict.values()):
        patch.set_facecolor(color)

    ax1.legend(zip(bp1['boxes']), ratios, fontsize="small", loc="upper right", title='Sampling Values')
    # ax1.set_xticks([1.5, 4.5, 7.5, 10.5, 13.5])
    ax1.set_xticklabels(ratios)
    ax1.set_yticks(range(0, 105, 20))
    ax1.set_xlabel('Sampling Frequence', fontsize="large")
    ax1.set_ylabel('Foraging (%)', fontsize="large")
    plt.tight_layout()

    maindir = '/tmp/swarm/data/experiments'
    fname = 'samplingboxplot'

    fig.savefig(
        maindir + '/' + fname + '.png')
    # pylint: disable = E1101

    plt.close(fig)


def compare_all_geese_efficiency():

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

    for j in range(len(suffixs)):
        data = read_data_sample_comparision(suffixs[j], simnames[j])

        medianf = np.quantile(data, 0.5, axis=0)
        q1f = np.quantile(data, 0.25, axis=0)
        q3f = np.quantile(data, 0.75, axis=0)


        xvalues = range(data.shape[1])
        ax1.plot(
            xvalues, medianf, # color=color[j],
            linewidth=1.0, label=suffixs[j])
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


def compare_sampling_differences_plot():
    # plt.style.use('fivethirtyeight')
    sampling_size = [0.1, 0.3, 0.5, 0.7, 0.9]

    datasnew_forge = [read_data_sample_ratio_geesebt(s) for s in sampling_size]
    datasold_forge = [read_data_sample_ratio_betrgeese(s) for s in sampling_size]
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
    plabels = ['Top Agents (GEESE-BT)', 'Parallel (BeTr-GEESE)', 'Top Agents (Complex-GEESE)']
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


def main():
    # fixed_behaviors_sampling_ratio()
    # compare_all_geese_efficiency()
    # compare_sampling_differences_plot()
    # read_data_sample_ratio_complex_geese(0.1)
    plot_restictive_grammar()


if __name__ == '__main__':
    main()