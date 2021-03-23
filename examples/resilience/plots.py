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


def plotgraph(folder):
    folders = pathlib.Path(folder).glob("1616*")
    # print(folders)
    flist = []
    data = []
    for f in folders:
        flist = [p for p in pathlib.Path(f).iterdir() if p.is_file()]
        _, _, d = np.genfromtxt(flist[0], autostrip=True, unpack=True, delimiter='|')
        data.append(d)
    # print(flist)
    data = np.array(data)
    print(data.shape)
    median = np.quantile(data, 0.5, axis=0)
    # print(median)
    q1 = np.quantile(data, 0.25, axis=0)
    q3 = np.quantile(data, 0.75, axis=0)
    print(median.shape, q1.shape, q3.shape)
    color = [
        'forestgreen', 'indianred',
        'gold', 'tomato', 'royalblue']
    colorshade = [
        'springgreen', 'lightcoral',
        'khaki', 'lightsalmon', 'deepskyblue']
    plt.style.use('fivethirtyeight')
    fig = plt.figure()
    ax1 = fig.add_subplot(1, 1, 1)
    xvalues = range(data.shape[1])
    ax1.plot(
        xvalues, median, color=color[0],
        linewidth=1.0)
    ax1.fill_between(
        xvalues, q3, q1,
        color=colorshade[0], alpha=0.3)
    plt.title('Foraging')
    # ax1.legend(title='$\it{m}$')
    ax1.set_xlabel('Steps')
    ax1.set_ylabel('Foraging %')

    # ax1.set_xticks(
    #     np.linspace(0, data.shape[-1], 5))
    plt.tight_layout()

    # fig.savefig(
    #     '/tmp/goal/data/experiments/' + pname + '.pdf')
    fig.savefig(
        folder + 'foraging' + '.png')
    plt.close(fig)


def main():
    plotgraph('/tmp/with')


# import os
# dname = os.path.join('/tmp', 'pygoal', 'data', 'experiments')
# pathlib.Path(dname).mkdir(parents=True, exist_ok=True)
# fname = os.path.join(dname, name + '.png')
# fig.savefig(fname)
# plt.close(fig)


if __name__ == '__main__':
    main()