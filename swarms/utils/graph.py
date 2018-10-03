"""Plot and Save the results."""

from matplotlib import pyplot as plt
import pandas as pd
import numpy as np

plt.style.use('fivethirtyeight')


class Graph:

    def __init__(self, directory, fname, fields):
        self.directory = directory
        self.fname = fname
        self.fields = fields
        self.data = self.load_file()
        self.mean = self.data[self.data['header'] == 'MEAN']
        self.std = self.data[self.data['header'] == 'STD']
        self.overall = self.data[
            self.data['header'] == 'OVERALL']['fitness'].values
        self.diverse = self.data[
            self.data['header'] == 'DIVERSE']['fitness'].values
        self.explore = self.data[
            self.data['header'] == 'EXPLORE']['fitness'].values
        self.forge = self.data[
            self.data['header'] == 'FORGE']['fitness'].values

    def gen_best_plots(self):
        fig = plt.figure()
        i = 1
        for field in self.fields:
            mean = self.mean[field].values
            std = self.std[field].values
            field_max = mean + std
            field_min = mean - std
            xvalues = range(1, len(mean) + 1)
            ax1 = fig.add_subplot(2, 1, i)
            i += 1
            # Plotting mean and standard deviation
            ax1.plot(xvalues, mean, color='blue', label='Mean ' + field)
            ax1.fill_between(
                xvalues, field_max, field_min, color='DodgerBlue', alpha=0.3)

            ax1.plot(xvalues, self.overall, color='red', label='Overall')
            ax1.plot(xvalues, self.diverse, color='green', label='Diversity')
            ax1.plot(xvalues, self.explore, color='orange', label='Explore')
            ax1.plot(xvalues, self.forge, color='indigo', label='Forge')
            plt.xlim(0, len(mean))
            ax1.set_xlabel('Steps')
            ax1.set_xlabel('Fitness')
            ax1.set_title('Fitness function')

        fig.savefig(self.directory + '/best.pdf')
        fig.savefig(self.directory + '/best.png')
        plt.close(fig)

    def load_file(self):
        data = pd.read_csv(self.directory + '/' + self.fname, sep='|')
        return data

    def save_step_graph(self, filename, fields):
        pass


class GraphACC:

    def __init__(self, directory, fname):
        self.directory = directory
        self.fname = fname
        self.data = self.load_file()
        self.step = self.data['step'].values
        self.performance = self.data['fitness'].values

    def gen_plot(self):
        fig = plt.figure()
        xvalues = self.step
        ax1 = fig.add_subplot(2, 1, 1)
        ax1.plot(xvalues, self.performance, color='red', label='Values')
        ax1.set_xlabel('Steps')
        ax1.set_xlabel('Performance')

        ax1.set_title('ACC Graph')

        fig.savefig(self.directory + '/acc.pdf')
        fig.savefig(self.directory + '/acc.png')
        plt.close(fig)

    def load_file(self):
        data = pd.read_csv(self.directory + '/' + self.fname, sep='|')
        return data

    def save_step_graph(self, filename, fields):
        pass


class ResMinMaxACC:

    def __init__(self, directory, fnames):
        self.directory = directory
        self.fnames = fnames

    def gen_plot(self):
        fig = plt.figure()

        self.normal_data = self.load_file(self.fnames[0])
        self.res1_data = self.load_file(self.fnames[1])
        self.res2_data = self.load_file(self.fnames[2])

        self.mean1 = np.nanmean(self.normal_data, axis=0)
        self.mean2 = np.nanmean(self.res1_data, axis=0)
        self.mean3 = np.nanmean(self.res2_data, axis=0)
        # print (self.mean1.shape, self.mean2.shape, self.mean2.shape)
        # self.sd = np.nanstd(self.data, axis=1)
        # self.max_sd = self.mean + self.sd
        # self.min_sd = self.mean - self.sd

        xvalues = range(1, self.mean1.shape[0] - 1)
        # print (xvalues)
        ax1 = fig.add_subplot(1, 1, 1)

        ax1.plot(xvalues, self.mean1[:-2], color='green', label='Normal')
        ax1.plot(xvalues, self.mean2[:-2], color='blue', label='Resilience 1')
        ax1.plot(xvalues, self.mean3[:-2], color='red', label='Resilience 2')
        # ax1.fill_between(
        # xvalues, self.min_sd, self.max_sd, color="red", alpha=0.3)

        ax1.set_xlabel('Iteration')
        ax1.set_xlabel('Fitness')

        ax1.set_title('ACC Graph with Resilience')

        fig.savefig(self.directory + '/acc_res.pdf')
        fig.savefig(self.directory + '/acc_res.png')
        plt.close(fig)

    def load_file(self, fname):
        # try:
        data = pd.read_csv(
            self.directory + '/' + fname, sep='|', skipinitialspace=True)
        return data
        # except FileNotFoundError:
        #    exit()

    def save_step_graph(self, filename, fields):
        pass


class PGraph:

    def __init__(self, directory, fnames):
        self.directory = directory
        self.fnames = fnames

    def gen_plot(self):
        fig = plt.figure()
        data = []
        for fname in self.fnames:
            if len(fname) > 1:
                values = self.load_file(fname)
                data.append(values['fitness'].tolist())

        data = np.array(data)

        self.mean = np.nanmean(data, axis=0)
        self.std = np.nanstd(data, axis=0)
        self.max_std = self.mean + self.std
        self.min_std = self.mean - self.std

        xvalues = range(1, self.mean.shape[0] - 1)

        ax1 = fig.add_subplot(1, 1, 1)

        ax1.plot(xvalues, self.mean[:-2], color='green', label='Mean')
        # ax1.plot(xvalues, self.std[:-2], color='red', label='STD')

        ax1.plot(
            xvalues, self.max_std[:-2], color='blue', label='Max',
            linestyle='dashed')
        ax1.plot(
            xvalues, self.min_std[:-2], color='purple', label='Min',
            linestyle='dashed')
        ax1.fill_between(
            xvalues, self.min_std[:-2], self.max_std[:-2], color="red",
            alpha=0.3)

        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Fitness')

        ax1.set_title('Performance')
        ax1.legend()
        fig.savefig(self.directory + '/average.pdf')
        fig.savefig(self.directory + '/average.png')
        plt.close(fig)

    def load_file(self, fname):
        try:
            data = pd.read_csv(
                fname, sep='|', skipinitialspace=True)
            return data
        except FileNotFoundError:
            exit()

    def save_step_graph(self, filename, fields):
        pass


class BoxGraph:

    def __init__(self, directory, fnames):
        self.directory = directory
        self.fnames = fnames

    def gen_plot(self):
        fig = plt.figure()
        data = []
        for fname in self.fnames:
            if len(fname) > 1:
                values = self.load_file(fname)
                data.append(values['fitness'].tolist())

        data = np.array(data)

        self.mean = np.nanmean(data, axis=0)
        self.std = np.nanstd(data, axis=0)
        self.max_std = self.mean + self.std
        self.min_std = self.mean - self.std

        maxgen = len(self.mean) - 2
        xvalues = range(1, maxgen + 1)

        ax1 = fig.add_subplot(1, 1, 1)

        ax1.plot(xvalues, self.mean[:-2], color='green', label='Mean')
        box_data = data.T
        box_data = [box_data[i] for i in range(500, maxgen, 500)]

        ax1.boxplot(
            box_data, 0, 'gD', positions=list(range(500, maxgen, 500)))

        ax1.fill_between(
            xvalues, self.min_std[:-2], self.max_std[:-2], color="red",
            alpha=0.3)

        plt.xlim(0, maxgen + 1)

        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Fitness')

        ax1.set_title('Performance')
        ax1.legend()
        fig.savefig(self.directory + '/boxplot.pdf')
        fig.savefig(self.directory + '/boxplot.png')
        plt.close(fig)

    def load_file(self, fname):
        try:
            data = pd.read_csv(
                fname, sep='|', skipinitialspace=True)
            return data
        except FileNotFoundError:
            exit()

    def save_step_graph(self, filename, fields):
        pass