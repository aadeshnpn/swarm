import os
import numpy as np
import scipy.stats as stats
import matplotlib

# If there is $DISPLAY, display the plot
if os.name == 'posix' and "DISPLAY" not in os.environ:
    matplotlib.use('Agg')

import matplotlib.pyplot as plt     # noqa: E402
import glob
import pathlib


def read_data_sample_ratio(ratio=0.1):
    maindir = '/tmp/swarm/data/experiments/'+ str(ratio)
    data = []
    # print(maindir)
    folders = pathlib.Path(maindir).glob("*CoevoSimulation")
    for f in folders:
        try:
            flist = [p for p in pathlib.Path(f).iterdir() if p.is_file() and p.match('simulation.csv')]
            # print(flist)
            _, _, d = np.genfromtxt(flist[0], autostrip=True, unpack=True, delimiter='|')
            # print(d.shape)
            data.append(d[-1])
        except:
            pass
    data = np.array(data)
    # print(ratio, data.shape, data)
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


def main():
    fixed_behaviors_sampling_ratio()


if __name__ == '__main__':
    main()