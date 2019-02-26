"""Script to draw performance graph for paper."""

from matplotlib import pyplot as plt
import numpy as np

plt.style.use('fivethirtyeight')


class Sampling:

    def __init__(self, directory, fnames, title="Sampling"):
        self.__dict__.update(locals())

    def gen_plot(self):
        fnames = ['random.npy', 'repeat.npy', 'diversity.npy']
        #fnames = [
        #    'foodonly.npy', 'exploreonly.npy', 'expfood.npy',
        #    'cfandfood.npy', 'expcf.npy', 'all.npy']
        data = []
        for fname in fnames:
            val = np.load(
                self.directory + '/' + str(fname))     # pylint: disable=E1101
            vallast = val[:, 4999]
            data.append(vallast)
        self.plot(data, fname)

    def plot(self, datas, rate):
        fig = plt.figure()
        ax1 = fig.add_subplot(1, 1, 1)
        plt.xlim(0, 6000)

        fig = plt.figure()

        ax1 = fig.add_subplot(1, 1, 1)

        data = np.array(datas)
        medianprops = dict(linewidth=2.5, color='firebrick')
        meanprops = dict(linewidth=2.5, color='#ff7f0e')
        # meanpointprops = dict(marker='D', markeredgecolor='black',
        #                    markerfacecolor='firebrick')
        bp1 = ax1.boxplot(
            data.T, 0, 'gD', showmeans=True, meanline=True,
            patch_artist=True, medianprops=medianprops, meanprops=meanprops)

        color_sequence = [
            '#1f77b4', '#aec7e8', '#9467bd', '#c7c7c7', '#2ca02c',
            '#98df8a', '#d62728', '#ff9896', '#1cafe2', '#c5b0d5',
            '#8c564b', '#c49c94', '#e377c2', '#f7b6d2', '#7f7f7f',
            '#ffbb78', '#bcbd22', '#dbdb8d', '#17becf', '#9edae5',
            '#9caae5', '#ff7f0e']

        color_sequence = ['blue', 'green', 'purple']

        for patch, color in zip(bp1['boxes'], color_sequence):
            patch.set_facecolor(color)

        ax1.set_xlabel('Fitness Function Type')
        ax1.set_ylabel('Foraging %')
        # ax1.set_xticklabels([])
        ax1.set_title(self.title)   # pylint: disable=E1101
        # sample_rates = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99]
        func = ['Random', 'Simplified', 'Diversity']
        #func = [
        #    '1. Task-Specific', '2. Explore', '3. Exploration + Task-Specific',
        #    '4. Prospective + Task-Specific',
        #    '5. Exploration + Prospective ', '6. All']
        ax1.legend(zip(bp1['boxes']), func, fontsize="small", loc="upper left")

        plt.tight_layout()
        fig.savefig(
            self.directory + '/' + 'fitness.pdf')    # pylint: disable=E1101
        fig.savefig(
            self.directory + '/' + 'fitness.png')    # pylint: disable=E1101
        plt.close(fig)


def main():
    """Parse args and call graph module."""
    fdir = "/home/aadeshnpn/Documents/BYU/hcmi/hri/ijcai/diversity_exp/numpy_diveristy"
    #fdir = "/home/aadeshnpn/Documents/BYU/hcmi/hri/ijcai/fitness_exp"
    graph = Sampling(fdir, None, "Single-Source Foraging")
    graph.gen_plot()


if __name__ == '__main__':
    main()
