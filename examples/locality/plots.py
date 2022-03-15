"""Plot file for locality experiments."""

import os
import numpy as np
import matplotlib

# If there is $DISPLAY, display the plot
if os.name == 'posix' and "DISPLAY" not in os.environ:
    matplotlib.use('Agg')

import matplotlib.pyplot as plt     # noqa: E402
from matplotlib.legend_handler import HandlerPathCollection


def plot_locality(locality, reversemap, begin=0, end=3000):
    simple_behavior_number = {
        'Explore': 1, 'CompositeSingleCarry': 2, 'CompositeDrop': 3,
        'MoveTowards': 4, 'MoveAway': 5
    }

    fig = plt.figure(figsize=(10, 6), dpi=300)
    ax1 = fig.add_subplot(1, 1, 1)
    plt.rcParams["legend.markerscale"] = 0.5
    ax1.spines['top'].set_color('none')
    ax1.spines['left'].set_position('zero')
    ax1.spines['right'].set_color('none')
    ax1.spines['bottom'].set_position('zero')
    # colors = []
    for j in range(1, 6):
        data = np.sum(np.squeeze(
                locality[begin:end, :, :, j]), axis=0)
        # print(i, np.sum(data))
        # c = ax1.pcolor(data)
        # fig.colorbar(c, ax=ax1)
        x, y = np.where(data >= 1)
        x1, y1 = zip(
            *[reversemap[(
                x[k], y[k])] for k in range(len(x))])
        ax1.scatter(
            x1, y1, s=data[x, y], alpha=0.5,
            label=list(
                simple_behavior_number.keys())[j-1])

    ax1.set_xticks(range(-50, 51, 10))
    ax1.set_yticks(range(-50, 51, 10))
    plt.xlim(-50, 50)
    plt.ylim(-50, 50)

    plt.legend(
        fontsize="small", bbox_to_anchor=(1.04, 1),
        borderaxespad=0, title='Primitive Behaviors')
    plt.tight_layout()
    maindir = '/tmp/swarm/data/experiments/locality'
    fname = 'locality_all_' + str(begin) + '_' + str(end) + '_'

    fig.savefig(
        maindir + '/' + fname + '.png')

    plt.close(fig)


def main():
    locality, reversemap = np.load(
        '/home/aadeshnpn/Desktop/coevolution/ANTS/locality.npy',
        allow_pickle=True)
    # plot_locality(
    #     locality, reversemap, begin=0, end=3000)
    plot_locality(
        locality, reversemap, begin=0, end=12000)


if __name__ == "__main__":
    main()
