"""Script to draw performance graph for paper."""

import sys
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from matplotlib.patches import Patch
plt.style.use('fivethirtyeight')


class ResMinMaxACC:

    def __init__(self, directory, fnames, title="ACC Graph with Resilience"):
        self.__dict__.update(locals())
        # self.directory = directory
        # self.fnames = fnames

    def gen_plot(self):
        fig = plt.figure()

        self.normal_data = self.load_file(
            self.fnames[0])   # pylint: disable=E1101
        self.res1_data = self.load_file(
            self.fnames[1])  # pylint: disable=E1101
        self.res2_data = self.load_file(
            self.fnames[2])  # pylint: disable=E1101

        # self.mean1 = np.nanmean(self.normal_data, axis=0)
        # self.mean2 = np.nanmean(self.res1_data, axis=0)
        # self.mean3 = np.nanmean(self.res2_data, axis=0)
        # print (self.mean1.shape, self.mean2.shape, self.mean2.shape)
        # self.sd = np.nanstd(self.data, axis=1)
        # self.max_sd = self.mean + self.sd
        # self.min_sd = self.mean - self.sd

        ax1 = fig.add_subplot(1, 1, 1)
        plt.xlim(0, 10000)
        plt.ylim(0, 100)
        box_data = self.normal_data.values.T
        box_data = [box_data[i] for i in range(0, 9000, 1000)]
        boxprops = dict(linewidth=1, color='blue')
        bp1 = ax1.boxplot(
            box_data, 0, whiskerprops=boxprops,
            showmeans=True, meanline=True, patch_artist=True,
             positions=range(0, 9000, 1000), widths=600)

        for patch in bp1['boxes']:
                patch.set_facecolor('blue')
                #patch.set_alpha(0.4)
                patch.set_edgecolor('blue') # or try 'black'
                patch.set_linewidth(1)

        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Performance')
        ax1.set_title('Ideal Sensors', fontsize=13)

        box_data = self.res1_data.values.T
        box_data = [box_data[i] for i in range(0, 9000, 1000)]
        fig1 = plt.figure()
        ax2 = fig1.add_subplot(1, 1, 1)
        bp2 = ax2.boxplot(
            box_data, 0, whiskerprops=boxprops,
            showmeans=True, meanline=True, patch_artist=True,
             positions=range(0, 9000, 1000), widths=450)

        for patch in bp2['boxes']:
                patch.set_facecolor('blue')
                #patch.set_alpha(0.4)
                patch.set_edgecolor('blue') # or try 'black'
                patch.set_linewidth(1)
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Performance')
        ax2.set_title('50% Sensor Failure', fontsize=13)

        fig.tight_layout()
        fig1.tight_layout()

        fig.savefig(self.directory + '/acc_res.pdf')    # pylint: disable=E1101
        fig.savefig(self.directory + '/acc_res.png')    # pylint: disable=E1101
        fig1.savefig(self.directory + '/acc_res1.pdf')    # pylint: disable=E1101
        fig1.savefig(self.directory + '/acc_res1.png')    # pylint: disable=E1101
        plt.close(fig)
        plt.close(fig1)

    def load_file(self, fname):
        # try:
        data = pd.read_csv(
            self.directory + '/' + fname, sep='|',  # pylint: disable=E1101
            skipinitialspace=True)
        return data
        # except FileNotFoundError:
        #    exit()

    def save_step_graph(self, filename, fields):
        pass


class ResMinMaxACC1:

    def __init__(self, directory, fnames, title="ACC Graph with Resilience"):
        self.__dict__.update(locals())
        # self.directory = directory
        # self.fnames = fnames

    def gen_plot(self):
        fig = plt.figure()

        self.normal_data = self.load_file(
            self.fnames[0])   # pylint: disable=E1101
        self.res1_data = self.load_file(
            self.fnames[1])  # pylint: disable=E1101
        self.res2_data = self.load_file(
            self.fnames[2])  # pylint: disable=E1101

        self.mean1 = np.nanmean(self.normal_data, axis=0)
        self.mean2 = np.nanmean(self.res1_data, axis=0)
        self.mean3 = np.nanmean(self.res2_data, axis=0)
        # print (self.mean1.shape, self.mean2.shape, self.mean2.shape)
        # self.sd = np.nanstd(self.data, axis=1)
        # self.max_sd = self.mean + self.sd
        # self.min_sd = self.mean - self.sd
        ax1 = fig.add_subplot(1, 1, 1)
        print (self.normal_data.shape)
        plt.xlim(0, 10000)
        box_data = self.normal_data.values.T
        box_data = [box_data[i] for i in range(1000, 9500, 1000)]
        boxprops = dict(linewidth=1, color='pink')
        bp1 = ax1.boxplot(
            box_data, 0, whiskerprops=boxprops,
            showmeans=True, meanline=True, patch_artist=True,
             positions=range(1000, 9500, 1000), widths=600)

        box_data = self.res1_data.values.T
        box_data = [box_data[i] for i in range(1000, 9500, 1000)]
        boxprops = dict(linewidth=1, color='lightblue')
        bp2 = ax1.boxplot(
            box_data, 0, whiskerprops=boxprops,
            showmeans=True, meanline=True, patch_artist=True,
             positions=range(1000, 9500, 1000), widths=450)

        box_data = self.res2_data.values.T
        box_data = [box_data[i] for i in range(1000, 9500, 1000)]
        boxprops = dict(linewidth=1, color='lightgreen')
        bp3 = ax1.boxplot(
            box_data, 0,whiskerprops=boxprops,
            showmeans=True, meanline=True, patch_artist=True,
             positions=range(1000, 9500, 1000), widths=250)

        for patch in bp1['boxes']:
                patch.set_facecolor('pink')
                patch.set_alpha(0.4)
                patch.set_edgecolor('pink') # or try 'black'
                patch.set_linewidth(1)

        for patch in bp2['boxes']:
                patch.set_facecolor('lightblue')
                patch.set_alpha(0.6)
                patch.set_edgecolor('lightblue') # or try 'black'
                patch.set_linewidth(1)

        for patch in bp3['boxes']:
                patch.set_facecolor('lightgreen')
                patch.set_alpha(0.6)
                patch.set_edgecolor('lightgreen') # or try 'black'
                patch.set_linewidth(1)

        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Performance')

        ax1.set_title('Single-Source Foraging\nActuator Failure')
        plt.tight_layout()
        ax1.legend(
            [bp1['boxes'][0], bp2['boxes'][0],bp3['boxes'][0]],
            ['Normal','Failure 1','Failure 2'], loc='upper left')
        fig.savefig(self.directory + '/acc_res.pdf')    # pylint: disable=E1101
        fig.savefig(self.directory + '/acc_res.png')    # pylint: disable=E1101
        plt.close(fig)

    def load_file(self, fname):
        # try:
        data = pd.read_csv(
            self.directory + '/' + fname, sep='|',  # pylint: disable=E1101
            skipinitialspace=True)
        return data
        # except FileNotFoundError:
        #    exit()

    def save_step_graph(self, filename, fields):
        pass


def main():
    """Parse args and call graph module."""
    filenames = sys.argv[1]
    fdir = sys.argv[2]
    filenames = filenames.split(',')

    # print (filenames)
    graph = ResMinMaxACC(fdir, filenames, "Single-Source Foraging")
    graph.gen_plot()

if __name__ == '__main__':
    main()
