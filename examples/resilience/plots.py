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


def plotgraph(n=100, agent='SimForgAgentWith', site='50-50'):
    # folders = pathlib.Path(folder).glob("1616*")
    # # print(folders)
    # flist = []
    # data = []
    # for f in folders:
    #     flist = [p for p in pathlib.Path(f).iterdir() if p.is_file()]
    #     _, _, d = np.genfromtxt(flist[0], autostrip=True, unpack=True, delimiter='|')
    #     data.append(d)
    # # print(flist)
    # data = np.array(data)
    # print(data.shape)

    # data = read_data_n_agent(n=n, agent=agent)
    dataf, datad = read_data_n_agent_site(n=n, agent=agent, site=site)
    # print(data.shape)
    medianf = np.quantile(dataf, 0.5, axis=0)
    q1f = np.quantile(dataf, 0.25, axis=0)
    q3f = np.quantile(dataf, 0.75, axis=0)

    mediand = np.quantile(datad, 0.5, axis=0)
    q1d = np.quantile(datad, 0.25, axis=0)
    q3d = np.quantile(datad, 0.75, axis=0)
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

    ax1.plot(
        xvalues, mediand, color=color[1],
        linewidth=1.0, label='Dead Agents')
    ax1.fill_between(
        xvalues, q3d, q1d,
        color=colorshade[1], alpha=0.3)
    plt.title('Foraging')
    # ax1.legend(title='$\it{m}$')
    ax1.set_xlabel('Steps')
    ax1.set_ylabel('%')

    # ax1.set_xticks(
    #     np.linspace(0, data.shape[-1], 5))
    plt.legend()
    plt.tight_layout()

    # fig.savefig(
    #     '/tmp/goal/data/experiments/' + pname + '.pdf')
    maindir = '/tmp/swarm/data/experiments/'
    nadir = os.path.join(maindir, str(n), agent)
    fig.savefig(
        nadir + str(site) + 'foraging' + '.png')
    plt.close(fig)


def read_data_n_agent(n=100, agent='SimForgAgentWith'):
    maindir = '/tmp/swarm/data/experiments/'
    nadir = os.path.join(maindir, str(n), agent)
    folders = pathlib.Path(nadir).glob("*ForagingSim*")
    flist = []
    data = []
    for f in folders:
        flist = [p for p in pathlib.Path(f).iterdir() if p.is_file()]
        _, _, d = np.genfromtxt(flist[0], autostrip=True, unpack=True, delimiter='|')
        data.append(d)
    data = np.array(data)
    return data


def read_data_n_agent_site(n=100, agent='SimForgAgentWith', site='5050'):
    maindir = '/tmp/swarm/data/experiments/'
    nadir = os.path.join(maindir, str(n), agent, site)
    folders = pathlib.Path(nadir).glob("*ForagingSim*")
    flist = []
    dataf = []
    datad = []
    for f in folders:
        flist = [p for p in pathlib.Path(f).iterdir() if p.is_file()]
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
    # print(dataf.shape, datad.shape)
    return dataf, datad



def boxplot(agent='SimForgAgentWith'):
    # data = [read_data_n_agent(n, agent)[:,-1] for n in [50, 100, 200, 300, 400]]
    data = [read_data_n_agent(n, agent)[:,-1] for n in [100, 200, 300, 400]]
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
    labels = [100, 200, 300, 400]
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
    ax1.legend(zip(bp1['boxes']), labels, fontsize="small", loc="upper right", title='no. of agents')
    ax1.set_xticklabels(labels)
    ax1.set_xlabel('No. of agents ')
    ax1.set_ylabel('Foraging Percentage')
    ax1.set_title('Swarm Foraging')

    plt.tight_layout()

    maindir = '/tmp/swarm/data/experiments/'
    fname = 'agentscomp' + agent

    fig.savefig(
        maindir + '/' + fname + '.png')
    # pylint: disable = E1101

    plt.close(fig)



def boxplotsiteloc(agent='SimForgAgentWith', site='5050'):
    # sites = ['-3030', '30-30', '3030', '-5050', '50-50', '5050', '-9090', '90-90']
    agents = [50, 100, 200, 300, 400]
    print(agent, site)
    dataf = [read_data_n_agent_site(n, agent, site=site)[0][:,-1] for n in agents]
    datad = [read_data_n_agent_site(n, agent, site=site)[1][:,-1] for n in agents]
    datadp = [(datad[i]/agents[i])*100 for i in range(len(agents))]
    fig = plt.figure()

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
    labels = ['50', '100', '200', '300', '400']
    medianprops = dict(linewidth=2.5, color='firebrick')
    meanprops = dict(linewidth=2.5, color='#ff7f0e')
    # data = [data[:, i] for i in range(4)]
    bp1 = ax1.boxplot(
        dataf, 0, 'gD', showmeans=True, meanline=True,
        patch_artist=True, medianprops=medianprops,
        meanprops=meanprops)
    for patch, color in zip(bp1['boxes'], colordict.values()):
        patch.set_facecolor(color)
    # plt.xlim(0, len(mean))
    ax1.legend(zip(bp1['boxes']), labels, fontsize="small", loc="upper right", title='Agent Size')
    ax1.set_xticklabels(labels)
    ax1.set_xlabel('Agent size')
    ax1.set_ylabel('Foraging Percentage')
    ax1.set_title('Swarm Foraging with distance '+ site[-2:])


    ax2 = fig.add_subplot(3, 1, 2)
    bp2 = ax2.boxplot(
        datad, 0, 'gD', showmeans=True, meanline=True,
        patch_artist=True, medianprops=medianprops,
        meanprops=meanprops)
    for patch, color in zip(bp2['boxes'], colordict.values()):
        patch.set_facecolor(color)
    # plt.xlim(0, len(mean))
    # ax2.legend(zip(bp2['boxes']), labels, fontsize="small", loc="upper right", title='Agent Size')
    ax2.set_xticklabels(labels)
    ax2.set_xlabel('Agent size')
    ax2.set_ylabel('No. Dead Agents')

    ax3 = fig.add_subplot(3, 1, 3)
    bp3 = ax3.boxplot(
        datadp, 0, 'gD', showmeans=True, meanline=True,
        patch_artist=True, medianprops=medianprops,
        meanprops=meanprops)
    for patch, color in zip(bp3['boxes'], colordict.values()):
        patch.set_facecolor(color)
    # plt.xlim(0, len(mean))
    # ax3.legend(zip(bp3['boxes']), labels, fontsize="small", loc="upper right", title='Agent Size')
    ax3.set_xticklabels(labels)
    ax3.set_xlabel('Agent size')
    ax3.set_ylabel('Dead Agents %')
    # ax2.set_title('Swarm Foraging with distance '+ site[-2:])
    plt.tight_layout()

    maindir = '/tmp/swarm/data/experiments/'
    # fname = 'agentsitecomp' + agent
    nadir = os.path.join(maindir, str(50))

    fig.savefig(
        nadir + agent + site +'agentsitecomp' + '.png')
    # fig.savefig(
    #     maindir + '/' + fname + '.png')
    # pylint: disable = E1101

    plt.close(fig)


def boxplotallsites(agent='SimForgAgentWith'):
    sites = ['-3131', '31-31', '3131', '-5151', '51-51', '5151', '-9191', '91-91']
    agents = [50, 100, 200, 300, 400]
    # print(agent, site)
    datasf = []
    datasd = []
    for n in agents:
        # print(site)
        dataf = [read_data_n_agent_site(n, agent, site=site)[0][:,-1] for site in sites]
        datad = [read_data_n_agent_site(n, agent, site=site)[1][:,-1] for site in sites]
        # print(n, np.hstack(dataf).shape, np.hstack(datad).shape)
        datasf.append(np.hstack(dataf))
        datasd.append(np.hstack(datad))
    fig = plt.figure()

    ax1 = fig.add_subplot(2, 1, 1)
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
    labels = ['50', '100', '200', '300', '400']
    medianprops = dict(linewidth=2.5, color='firebrick')
    meanprops = dict(linewidth=2.5, color='#ff7f0e')
    # data = [data[:, i] for i in range(4)]
    bp1 = ax1.boxplot(
        datasf, 0, 'gD', showmeans=True, meanline=True,
        patch_artist=True, medianprops=medianprops,
        meanprops=meanprops)
    for patch, color in zip(bp1['boxes'], colordict.values()):
        patch.set_facecolor(color)
    # plt.xlim(0, len(mean))
    ax1.legend(zip(bp1['boxes']), labels, fontsize="small", loc="upper right", title='Agent Size')
    ax1.set_xticklabels(labels)
    ax1.set_xlabel('Agent size')
    ax1.set_ylabel('Foraging Percentage')
    # ax1.set_title('Swarm Foraging with distance '+ site[-2:])


    ax2 = fig.add_subplot(2, 1, 2)
    bp2 = ax2.boxplot(
        datasd, 0, 'gD', showmeans=True, meanline=True,
        patch_artist=True, medianprops=medianprops,
        meanprops=meanprops)
    for patch, color in zip(bp2['boxes'], colordict.values()):
        patch.set_facecolor(color)
    # plt.xlim(0, len(mean))
    ax2.legend(zip(bp2['boxes']), labels, fontsize="small", loc="upper right", title='Agent Size')
    ax2.set_xticklabels(labels)
    ax2.set_xlabel('Agent size')
    ax2.set_ylabel('No. Dead Agents')
    # ax2.set_title('Swarm Foraging with distance '+ site[-2:])
    plt.tight_layout()

    maindir = '/tmp/swarm/data/experiments/'
    # fname = 'agentsitecomp' + agent
    nadir = os.path.join(maindir, str(50), agent)

    fig.savefig(
        nadir + 'agentallsitecomp' + '.png')
    # fig.savefig(
    #     maindir + '/' + fname + '.png')
    # pylint: disable = E1101

    plt.close(fig)


def main():
    # agents = [50, 100, 200, 300, 400]
    atype = ['SimForgAgentWith', 'SimForgAgentWithout']
    # boxplotsiteloc(atype[1])
    # boxplot(atype[1])
    sitelocation  = [
        {"x":51, "y":-51, "radius":10, "q_value":0.9},
        {"x":51, "y":51, "radius":10, "q_value":0.9},
        {"x":-51, "y":51, "radius":10, "q_value":0.9},
        {"x":31, "y":-31, "radius":10, "q_value":0.9},
        {"x":31, "y":31, "radius":10, "q_value":0.9},
        {"x":-31, "y":31, "radius":10, "q_value":0.9},
        {"x":91, "y":-91, "radius":10, "q_value":0.9},
        {"x":-91, "y":91, "radius":10, "q_value":0.9},
    ]
    i = 7
    # sitename = str(sitelocation[i]['x']) + str(sitelocation[i]['y'])
    # print(sitename)

    # for i in range(len(sitelocation)):
    #     sitename = str(sitelocation[i]['x']) + str(sitelocation[i]['y'])
    #     for n in [50, 100, 200, 300, 400]:
    #         plotgraph(n=n, agent=atype[1], site=sitename)
    #         plotgraph(n=n, agent=atype[0], site=sitename)
    #         print(sitename, n)


    for i in range(len(sitelocation)):
        sitename = str(sitelocation[i]['x']) + str(sitelocation[i]['y'])
        for t in atype:
            boxplotsiteloc(agent=t, site=sitename)

    # for a in atype:
    #     boxplotallsites(a)
# import os
# dname = os.path.join('/tmp', 'pygoal', 'data', 'experiments')
# pathlib.Path(dname).mkdir(parents=True, exist_ok=True)
# fname = os.path.join(dname, name + '.png')
# fig.savefig(fname)
# plt.close(fig)


if __name__ == '__main__':
    main()