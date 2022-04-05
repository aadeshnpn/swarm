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
    # ax1.set_xlabel(list(range(-50, 51, 10)), fontsize="large")
    # ax1.set_ylabel(list(range(-50, 51, 10)), fontsize="large")
    plt.xlim(-50, 50)
    plt.ylim(-50, 50)

    plt.legend(
        fontsize="large", bbox_to_anchor=(1.04, 1),
        borderaxespad=0)
    plt.tight_layout()
    maindir = '/tmp/swarm/data/experiments/locality'
    fname = 'locality_all_' + str(begin) + '_' + str(end) + '_'

    fig.savefig(
        maindir + '/' + fname + '.png')

    plt.close(fig)


def plot_locality_gif(locality, reversemap, frames=100):
    simple_behavior_number = {
        'Explore': 1, 'CompositeSingleCarry': 2, 'CompositeDrop': 3,
        'MoveTowards': 4, 'MoveAway': 5
    }
    imgno = 1
    for i in range(50, 12000-frames, frames):
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
                    locality[0:i, :, :, j]), axis=0)
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
        # ax1.set_xlabel(list(range(-50, 51, 10)), fontsize="large")
        # ax1.set_ylabel(list(range(-50, 51, 10)), fontsize="large")
        plt.xlim(-50, 50)
        plt.ylim(-50, 50)

        # plt.legend(
        #     fontsize="large", bbox_to_anchor=(1.04, 1),
        #     borderaxespad=0)
        plt.tight_layout()
        maindir = '/tmp/swarm/data/experiments/locality'
        fname = 'locality-' + str(imgno)

        fig.savefig(
            maindir + '/' + fname + '.png')
        imgno += 1
        plt.close(fig)
        # ffmpeg -framerate 1 -i locality-%00d.png -c:v libx264 -profile:v high -crf 20 -pix_fmt yuv420p /tmp/output.mp4


def plot_foraging_gif(agents, reversemap, static_objs, frames=100):
    simple_behavior_number = {
        'Explore': 1, 'CompositeSingleCarry': 2, 'CompositeDrop': 3,
        'MoveTowards': 4, 'MoveAway': 5, 'WithFood': 6
    }
    imgno = 1
    for i in range(1, 100-frames, frames):
        fig = plt.figure(figsize=(10, 6), dpi=300)
        ax1 = fig.add_subplot(1, 1, 1)
        plt.rcParams["legend.markerscale"] = 0.5
        # ax1.spines['top'].set_color('none')
        # ax1.spines['left'].set_position('zero')
        # ax1.spines['right'].set_color('none')
        # ax1.spines['bottom'].set_position('zero')
        # colors = []
        colors = [
            '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
            '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
        a, x, y, _ = np.where(agents[i] > 0)
        agent_state = np.squeeze(agents[i])[a, x, y]

        x1, y1 = zip(
            *[reversemap[(
                x[k], y[k])] for k in range(len(x))])

        state6 = agent_state >= 100
        agent_state[state6] = agent_state[state6]-100
        state1 = agent_state == 1
        state2 = agent_state == 2
        state3 = agent_state == 3
        state4 = agent_state == 4
        state5 = agent_state == 5
        states = [state1, state2, state3, state4, state5, state6]
        markers = ['*'] * 5 + ['D']
        for j in range(6):
            ax1.scatter(
                np.array(x1)[states[j]], np.array(y1)[states[j]], c=colors[j],
                alpha=0.5, marker=markers[j], label=list(
                    simple_behavior_number.keys())[j])

        # Plot hub, site, and obstacles
        sites = static_objs[i][0]
        hubs = static_objs[i][1]
        # print(sites, hubs)
        ax1.scatter(
            [-30], [30], c=colors[6], s=sites*10,
            alpha=0.5, marker='8', label='Site')
        ax1.scatter(
            [0], [0], c=colors[7], s=100+hubs*2,
            alpha=0.5, marker='s', label='Hub')
        # Obstacles
        obs_locs = [(-30, -30), (30, 30), (30, -30), (-15, 0)]
        for oloc in obs_locs:
            ax1.scatter(
                [oloc[0]], [oloc[1]], c=colors[8], s=300,
                alpha=0.5, marker='X', label='Obstacle')

        ax1.set_xticks([])
        ax1.set_yticks([])

        # ax1.set_xticks(range(-50, 51, 10))
        # ax1.set_yticks(range(-50, 51, 10))
        # ax1.set_xlabel(list(range(-50, 51, 10)), fontsize="large")
        # ax1.set_ylabel(list(range(-50, 51, 10)), fontsize="large")
        plt.xlim(-50, 50)
        plt.ylim(-50, 50)

        plt.legend(
            fontsize="large", bbox_to_anchor=(1.04, 1),
            borderaxespad=0)
        plt.tight_layout()
        maindir = '/tmp/swarm/data/experiments/locality'
        fname = 'foraging-' + str(imgno)

        fig.savefig(
            maindir + '/' + fname + '.png')
        imgno += 1
        plt.close(fig)
        # ffmpeg -framerate 1 -i locality-%00d.png -c:v libx264 -profile:v high -crf 20 -pix_fmt yuv420p /tmp/output.mp4


def main():
    # locality, reversemap = np.load(
    #     '/home/aadeshnpn/Desktop/coevolution/ANTS/locality.npy',
    #     allow_pickle=True)

    agents, reversemap = np.load(
        '/tmp/visual.npy',
        allow_pickle=True)
    static_objs = np.load(
        '/tmp/staticobjs.npy',
        allow_pickle=True)
    # plot_locality(
    #     locality, reversemap, begin=0, end=3000)
    # plot_locality(
    #     locality, reversemap, begin=0, end=2000)

    # plot_locality_gif(locality, reversemap, frames=50)
    plot_foraging_gif(agents, reversemap, static_objs, frames=1)


if __name__ == "__main__":
    main()
