"""Script to draw performance graph for paper."""

from matplotlib import pyplot as plt
import numpy as np

plt.style.use('fivethirtyeight')


class BoxPlotSideBySide:

    def __init__(self, directory, ylimit=(-1, 110), title="Sampling"):
        self.__dict__.update(locals())

    def gen_plot(self):
        # fnames = ['random.npy', 'repeat.npy', 'diversity.npy']
        fnames = ['datae.obj.npy', 'data.obj.npy']
        data = []
        for fname in fnames:
            val = np.load(
                self.directory + '/' + str(fname))     # pylint: disable=E1101
            # vallast = val[:, 4999]
            data.append(val)
        self.plot(data, fname)

    def plot(self, datas, rate):
        fig = plt.figure()
        i = 0
        ax1 = fig.add_subplot(1, 2, 1)
        ax2 = fig.add_subplot(1, 2, 2, sharex=ax1, sharey=ax1)
        ax = [ax1, ax2]
        title = ['Evolved', 'Handcoded']
        for data in datas:
            maxgen = 5000
            box_data = data.T
            box_data = [box_data[i] for i in range(0, maxgen, 1000)]

            ax[i].boxplot(
                box_data, 0, 'gD', positions=list(range(0, maxgen, 1000)),
                widths=200)
            ax[i].tick_params(axis='both', which='major', labelsize=10)
            ax[0].set_xlabel('Iteration', fontsize="medium")
            ax[0].set_ylabel('Performance', fontsize="medium")
            ax[i].set_title(title[i], fontsize='medium')   # pylint: disable=E1101
            i += 1
        ax1.get_shared_x_axes().join(ax1, ax2)
        plt.tight_layout()
        plt.xlim(0, maxgen + 1)
        plt.ylim(self.ylimit[0], self.ylimit[1])    # pylint: disable=E1101
        # fig.suptitle('Singe-source Foraging', fontsize="medium")
        fig.savefig(self.directory + '/boxplotcomp.pdf')    # pylint: disable=E1101
        fig.savefig(self.directory + '/boxplotcomp.png')    # pylint: disable=E1101
        # plt.xlabel("Iteration", fontsize='medium')
        # plt.ylabel("Performance", fontsize='medium')
        plt.close(fig)


class BoxPlotCompSideBySide:

    def __init__(self, directory, ylimit=(-1, 110), title="Sampling"):
        self.__dict__.update(locals())

    def gen_plot(self):
        # fnames = ['random.npy', 'repeat.npy', 'diversity.npy']
        fnames = [
            'sfe.obj.npy', 'sfh.obj.npy',
            'cte.obj.npy', 'cth.obj.npy',
            'nme.obj.npy', 'nmh.obj.npy']
        data = []
        for fname in fnames:
            val = np.load(
                self.directory + '/' + str(fname))     # pylint: disable=E1101
            vallast = val[:, 4999]
            data.append(vallast)
        self.plot(data, fname)

    def plot(self, datas, rate):
        fig = plt.figure()
        ax1 = fig.add_subplot(1, 3, 1)
        ax2 = fig.add_subplot(1, 3, 2, sharex=ax1, sharey=ax1)
        ax3 = fig.add_subplot(1, 3, 3, sharex=ax1, sharey=ax1)
        ax = [ax1, ax2, ax3]
        # title = ['Evolved', 'Handcoded']
        color_sequence = [
            '#1f77b4', '#aec7e8', '#9467bd', '#c7c7c7', '#2ca02c',
            '#98df8a', '#d62728', '#ff9896', '#1cafe2', '#c5b0d5',
            '#8c564b', '#c49c94', '#e377c2', '#f7b6d2', '#7f7f7f',
            '#ffbb78', '#bcbd22', '#dbdb8d', '#17becf', '#9edae5',
            '#9caae5', '#ff7f0e']

        forge = datas[:2]
        forge = np.array(forge)
        medianprops = dict(linewidth=2, color='firebrick')
        meanprops = dict(linewidth=2, color='#ff7f0e')
        bp1 = ax[0].boxplot(
            forge.T, 0, 'gD', showmeans=True, meanline=True,
            patch_artist=True, medianprops=medianprops, meanprops=meanprops, widths=0.5)

        for patch, color in zip(bp1['boxes'], color_sequence):
            patch.set_facecolor(color)

        ax[0].tick_params(axis='both', which='major', labelsize=10)
        ax[0].set_xlabel('Iteration', fontsize="small")
        ax[0].set_ylabel('Performance', fontsize="small")
        ax[0].set_title('Foraging', fontsize='small')   # pylint: disable=E1101

        func = [
            '1. Evolved', '2. Handcoded'
            ]
        ax1.legend(zip(bp1['boxes']), func, fontsize="small", loc="upper left")

        ct = datas[2:4]
        ct = np.array(ct)
        bp2 = ax[1].boxplot(
            ct.T, 0, 'gD', showmeans=True, meanline=True, patch_artist=True,
            medianprops=medianprops, meanprops=meanprops, widths=0.5)

        for patch, color in zip(bp2['boxes'], color_sequence):
            patch.set_facecolor(color)
        ax[1].set_title('Cooperative Transport', fontsize='small')   # pylint: disable=E1101

        nm = datas[4:]
        nm = np.array(nm)
        bp3 = ax[2].boxplot(
            nm.T, 0, 'gD', showmeans=True, meanline=True, patch_artist=True,
            medianprops=medianprops, meanprops=meanprops, widths=0.5)

        for patch, color in zip(bp3['boxes'], color_sequence):
            patch.set_facecolor(color)
        ax[2].set_title('Nest Maintenance', fontsize='small')   # pylint: disable=E1101

        ax1.get_shared_x_axes().join(ax1, ax2, ax3)
        plt.tight_layout()
        # plt.xlim(0, maxgen + 1)
        # plt.ylim(self.ylimit[0], self.ylimit[1])    # pylint: disable=E1101
        # fig.suptitle('Singe-source Foraging', fontsize="medium")
        fig.savefig(self.directory + '/boxplotcomp.pdf')    # pylint: disable=E1101
        fig.savefig(self.directory + '/boxplotcomp.png')    # pylint: disable=E1101
        # plt.xlabel("Iteration", fontsize='medium')
        # plt.ylabel("Performance", fontsize='medium')
        plt.close(fig)


def main():
    """Parse args and call graph module."""
    fdir = "/home/aadeshnpn/Documents/BYU/hcmi/hri/ijcai/all"
    # graph = BoxPlotSideBySide(fdir, title="Single-Source Foraging")
    graph = BoxPlotCompSideBySide(fdir, title="Single-Source Foraging")
    graph.gen_plot()


if __name__ == '__main__':
    main()
