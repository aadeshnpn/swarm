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


def plotgraph(n=100):

    # data = read_data_n_agent(n=n, agent=agent)
    # dataf = read_data_n_agent(n=n, filename=fname)
    dataf =  read_data()
    print(dataf.shape)
    medianf = np.quantile(dataf, 0.5, axis=0)
    q1f = np.quantile(dataf, 0.25, axis=0)
    q3f = np.quantile(dataf, 0.75, axis=0)

    # mediand = np.quantile(datad, 0.5, axis=0)
    # q1d = np.quantile(datad, 0.25, axis=0)
    # q3d = np.quantile(datad, 0.75, axis=0)
    # print(median.shape, q1.shape, q3.shape)
    color = [
        'forestgreen', 'indianred',
        'gold', 'tomato', 'royalblue']
    colorshade = [
        'springgreen', 'lightcoral',
        'khaki', 'lightsalmon', 'deepskyblue']
    plt.style.use('fivethirtyeight')
    fig = plt.figure()
    ax1 = fig.add_subplot(1, 1, 1)
    xvalues = range(dataf.shape[1])
    ax1.plot(
        xvalues, medianf, color=color[0],
        linewidth=1.0, label='Food')
    ax1.fill_between(
        xvalues, q3f, q1f,
        color=colorshade[0], alpha=0.3)


    ax1.set_xlabel('Steps')
    ax1.set_ylabel('%')

    # ax1.set_xticks(
    #     np.linspace(0, data.shape[-1], 5))
    plt.legend()
    plt.tight_layout()

    # fig.savefig(
    #     '/tmp/goal/data/experiments/' + pname + '.pdf')
    maindir = '/tmp/swarm/data/experiments/'
    # nadir = os.path.join(maindir, str(n), agent)
    fig.savefig(
        maindir + label + '.png')
    plt.close(fig)


def boxplot(fname='/tmp/old.txt'):
    datao = read_data_n_agent(n=100, filename='/tmp/old.txt')[:,-1]
    datan = read_data_n_agent(n=100, filename='/tmp/new.txt')[:,-1]
    data = [datao, datan]
    fig = plt.figure()

    ax1 = fig.add_subplot(1, 1, 1)
    colordict = {
        0: 'forestgreen',
        1: 'indianred',
        2: 'gold',
        3: 'tomato',
        4: 'royalblue'}
    colorshade = [
        'springgreen', 'lightcoral',
        'khaki', 'lightsalmon', 'deepskyblue']
    # colordict = {
    #     0: 'bisque',
    #     1: 'darkorange',
    #     2: 'orangered',
    #     3: 'seagreen'
    # }

    # labels = ['Agent-Key', 'Key-Door', 'Door-Goal', 'Total']
    # labels = [50, 100, 200, 300, 400]
    labels = ['Old BNF', 'New BNF']
    medianprops = dict(linewidth=2.5, color='firebrick')
    meanprops = dict(linewidth=2.5, color='#ff7f0e')
    # data = [data[:, i] for i in range(4)]
    bp1 = ax1.boxplot(
        data, 0, 'gD', showmeans=True, meanline=True,
        patch_artist=True, medianprops=medianprops,
        meanprops=meanprops)
    for patch, color in zip(bp1['boxes'], colordict.values()):
        patch.set_facecolor(color)
    # plt.xlim(0, len(mean))
    ax1.legend(zip(bp1['boxes']), labels, fontsize="small", loc="upper right", title='BNF Type')
    ax1.set_xticklabels(labels)
    ax1.set_xlabel('BNF Grammar Type')
    ax1.set_ylabel('Foraging Percentage')
    ax1.set_title('Swarm Foraging Evolved Behaviors')

    plt.tight_layout()

    maindir = '/tmp/swarm/data/experiments/'
    fname = 'agentscomp'

    fig.savefig(
        maindir + '/' + fname + '.png')
    # pylint: disable = E1101

    plt.close(fig)


def read_data_n_agent(n=100, filename='/tmp/old.txt'):
    maindir = '/tmp/swarm/data/experiments/'
    files = np.genfromtxt(filename, unpack=True, autostrip=True, dtype=np.str)
    data = []
    for f in files:
        # print(f)
        _, _, d = np.genfromtxt(str(f), autostrip=True, unpack=True, delimiter='|')
        data.append(d)
    data = np.array(data)
    # print(data.shape)
    return data


def read_data():
    maindir = '/tmp/swarm/data/experiments/'
    # nadir = os.path.join(maindir, str(n), agent)
    folders = pathlib.Path(maindir).glob("*" + "ValidateSForgeNewPPAComm1")
    flist = []
    data = []
    for f in folders:
        flist = [p for p in pathlib.Path(f).iterdir() if p.is_file() and p.match('simulation.csv')]
        # print(flist)
        _, _, d = np.genfromtxt(flist[0], autostrip=True, unpack=True, delimiter='|')
        data.append(d)
    data = np.array(data)
    # print(data.shape)
    return data


def read_data_n_agent_site(n=100, i=5000):
    maindir = '/tmp/plots/data/experiments/'
    # maindir = '/home/aadeshnpn/Desktop/evolved_ppa/experiments/'
    nadir = os.path.join(maindir, str(n), str(i))
    folders = pathlib.Path(nadir).glob("*TestSForgeNewPPAComm*")
    flist = []
    dataf = []
    datad = []
    for f in folders:
        flist = [p for p in pathlib.Path(f).iterdir() if p.is_file() and p.match('simulation.csv')]
        try:
            # print(flist)
            _, _, f, d = np.genfromtxt(flist[0], autostrip=True, unpack=True, delimiter='|')
            dataf.append(f)
            datad.append(d)
        except IndexError:
            pass
    # print(data)
    dataf = np.array(dataf)
    datad = np.array(datad)
    print(dataf.shape, datad.shape, n)
    return dataf, datad


def boxplotsiteloc():
    # sites = ['-3030', '30-30', '3030', '-5050', '50-50', '5050', '-9090', '90-90']
    agents = [50, 100, 200] #, 300, 400, 500]
    # agents = [100]
    # print(agent, site)
    dataf = [read_data_n_agent_site(n)[0][:,-1] for n in agents]
    datad = [read_data_n_agent_site(n)[1][:,-1] for n in agents]
    datadp = [(datad[i]/agents[i])*100 for i in range(len(agents))]
    fig = plt.figure(figsize=(6, 8), dpi=100)

    ax1 = fig.add_subplot(3, 1, 1)
    colordict = {
        0: 'forestgreen',
        1: 'indianred',
        2: 'gold',
        3: 'tomato',
        4: 'royalblue',
        5: 'orchid',
        6: 'olivedrab',
        7: 'peru',
        8: 'linen'}
    colorshade = [
        'springgreen', 'lightcoral',
        'khaki', 'lightsalmon', 'deepskyblue']
    # colordict = {
    #     0: 'bisque',
    #     1: 'darkorange',
    #     2: 'orangered',
    #     3: 'seagreen'
    # }

    # labels = ['Agent-Key', 'Key-Door', 'Door-Goal', 'Total']
    # labels = ['30', '30', '30', '50', '50', '50', '90', '90']
    labels = ['50', '100', '200']  #, '300', '400', '500']
    medianprops = dict(linewidth=1.5, color='firebrick')
    meanprops = dict(linewidth=1.5, color='#ff7f0e')
    # data = [data[:, i] for i in range(4)]
    bp1 = ax1.boxplot(
        dataf, 0, 'gD', showmeans=True, meanline=True,
        patch_artist=True, medianprops=medianprops,
        meanprops=meanprops)
    for patch, color in zip(bp1['boxes'], colordict.values()):
        patch.set_facecolor(color)
    # plt.xlim(0, len(mean))
    ax1.legend(zip(bp1['boxes']), labels, fontsize="medium", loc="lower right", title='Agent Size')
    ax1.set_xticklabels(labels)
    ax1.set_yticks(range(0, 105, 20))
    # ax1.set_xlabel('Agent size')
    ax1.set_ylabel('Foraging Percentage', fontsize="large")
    ax1.set_title('Swarm Foraging with distance '+ str(50))

    ax2 = fig.add_subplot(3, 1, 2)
    bp2 = ax2.boxplot(
        datad, 0, 'gD', showmeans=True, meanline=True,
        patch_artist=True, medianprops=medianprops,
        meanprops=meanprops)
    for patch, color in zip(bp2['boxes'], colordict.values()):
        patch.set_facecolor(color)
    # plt.xlim(0, len(mean))
    # ax2.legend(zip(bp2['boxes']), labels, fontsize="small", loc="upper right", title='Agent Size')
    ax2.set_yticks(range(0,105, 20))
    ax2.set_xticklabels(labels)
    # ax2.set_xlabel('Agent size')
    ax2.set_ylabel('No. Dead Agents', fontsize="large")

    ax3 = fig.add_subplot(3, 1, 3)
    bp3 = ax3.boxplot(
        datadp, 0, 'gD', showmeans=True, meanline=True,
        patch_artist=True, medianprops=medianprops,
        meanprops=meanprops)
    for patch, color in zip(bp3['boxes'], colordict.values()):
        patch.set_facecolor(color)
    # plt.xlim(0, len(mean))
    # ax3.legend(zip(bp3['boxes']), labels, fontsize="small", loc="upper right", title='Agent Size')
    ax3.set_yticks(range(0,105, 20))
    ax3.set_xticklabels(labels)
    ax3.set_xlabel('Agent size', fontsize="large")
    ax3.set_ylabel('Dead Agents %', fontsize="large")
    # ax2.set_title('Swarm Foraging with distance '+ site[-2:])
    # plt.tight_layout()

    maindir = '/tmp/swarm/data/experiments/'
    # fname = 'agentsitecomp' + agent
    nadir = os.path.join(maindir, str(50))

    fig.savefig(
        nadir + 'agentsitecomp' + '.png')
    # fig.savefig(
    #     maindir + '/' + fname + '.png')
    # pylint: disable = E1101

    plt.close(fig)


def main():
    # read_data_n_agent()
    # plotgraph()
    # boxplot()
    # read_data()
    boxplotsiteloc()


if __name__ == '__main__':
    main()