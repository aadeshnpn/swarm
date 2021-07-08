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


def plotgraph(n=100, fname='/tmp/old.txt', label='Old BNF Grammar'):

    # data = read_data_n_agent(n=n, agent=agent)
    dataf = read_data_n_agent(n=n, filename=fname)
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


def plotgraphsite(n=100, agent='SimForgAgentWith', site='51-51'):
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


def read_data_n_agent(n=100, agent='ExecutingAgent'):
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


def read_data_n_agent_site(n=100, agent='ExecutingAgent', site='20'):
    maindir = '/tmp/results_site_distance/experiments/'
    # maindir = '/home/aadeshnpn/Desktop/evolved_ppa/experiments/'
    nadir = os.path.join(maindir, str(n), agent, str(site))
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
    print(dataf.shape, datad.shape, n)
    return dataf, datad


def boxplotsiteloc(agent='ExecutingAgent', site='51-51'):
    # sites = ['-3030', '30-30', '3030', '-5050', '50-50', '5050', '-9090', '90-90']
    agents = [50, 100, 200, 300, 400, 500]
    # agents = [100]
    print(agent, site)
    dataf = [read_data_n_agent_site(n, agent, site=site)[0][:,-1] for n in agents]
    datad = [read_data_n_agent_site(n, agent, site=site)[1][:,-1] for n in agents]
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
    labels = ['50', '100', '200', '300', '400', '500']
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
    ax1.set_title('Swarm Foraging with distance '+ str(int(site[-2:])-1))

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
    ax3.set_ylabel('Dead Agents (%)', fontsize="large")
    # ax2.set_title('Swarm Foraging with distance '+ site[-2:])
    # plt.tight_layout()

    maindir = '/tmp/swarm/data/experiments/'
    # fname = 'agentsitecomp' + agent
    nadir = os.path.join(maindir, str(50))

    fig.savefig(
        nadir + agent + site +'agentsitecomp' + '.png')
    # fig.savefig(
    #     maindir + '/' + fname + '.png')
    # pylint: disable = E1101

    plt.close(fig)


def boxplotallsites(agent='ExecutingAgent'):
    sites = ['-3131', '31-31', '3131', '-5151', '51-51', '5151', '-9191', '91-91']
    agents = [50, 100, 200, 300, 400]
    # print(agent, site)
    datasf = []
    datasd = []
    datasdp = []
    for n in agents:
        # print(site)
        dataf = [read_data_n_agent_site(n, agent, site=site)[0][:,-1] for site in sites]
        datad = [read_data_n_agent_site(n, agent, site=site)[1][:,-1] for site in sites]
        datadp = [(d/n)*100.0 for d in datad]
        # print(n, np.hstack(dataf).shape, np.hstack(datad).shape)
        datasf.append(np.hstack(dataf))
        datasd.append(np.hstack(datad))
        datasdp.append(np.hstack(datadp))
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


    ax2 = fig.add_subplot(3, 1, 2)
    bp2 = ax2.boxplot(
        datasd, 0, 'gD', showmeans=True, meanline=True,
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
        datasdp, 0, 'gD', showmeans=True, meanline=True,
        patch_artist=True, medianprops=medianprops,
        meanprops=meanprops)
    for patch, color in zip(bp3['boxes'], colordict.values()):
        patch.set_facecolor(color)
    # plt.xlim(0, len(mean))
    # ax3.legend(zip(bp3['boxes']), labels, fontsize="small", loc="upper right", title='Agent Size')
    ax3.set_xticklabels(labels)
    ax3.set_xlabel('Agent size')
    ax3.set_ylabel('Dead Agents (%)')
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


def boxplotallsitesdist(agent='ExecutingAgent'):
    plt.style.use('fivethirtyeight')
    # sites = ['-3131', '31-31', '3131', '-5151', '51-51', '5151', '-9191', '91-91']
    # sites = ['-3131', '31-31', '3131', '-5151', '5151', '-9191', '91-91']
    sites = [20, 25, 30, 40, 50, 60, 70, 80, 90]
    agents = [100] #, 200, 300, 400]
    # print(agent, site)
    datasf = []
    # datasd = []
    # datasdp = []
    for k in sites:
        # print(site)
        dataf = read_data_n_agent_site(n=100, agent=agent, site=k)[0][:,-1]
        # print(k, dataf.shape)
        # datad = [read_data_n_agent_site(n=100, agent=agent, site=k)[1][:,-1] ]
        # datadp = [(d/n)*100.0 for d in datad]
        # print(n, np.hstack(dataf).shape, np.hstack(datad).shape)
        datasf.append(dataf)
        # datasd.append(np.hstack(datad))
        # datasdp.append(np.hstack(datadp))
    fig = plt.figure()

    ax1 = fig.add_subplot(1, 1, 1)
    colordict = {
        0: 'forestgreen',
        1: 'peru',
        2: 'royalblue',
        3: 'orchid',
        4: 'olivedrab',
        5: 'gold',
        6: 'linen',
        7: 'indianred',
        8: 'tomato'}
    colorshade = [
        'springgreen', 'lightcoral',
        'khaki', 'lightsalmon', 'deepskyblue']
    # colordict = {
    #     0: 'bisque',
    #     1: 'darkorange',
    #     2: 'orangered',
    #     3: 'seagreen'
    # }

    labels = [str(a) for a in sites]
    # xlabels = ['1', '2', '3', '4', '5', '6', '7', '8']
    # labels = [labels[i]+ ' / ' + xlabels[i] for i in range(len(labels))]
    # labels = ['50', '100'] #, '200', '300', '400']
    medianprops = dict(linewidth=1.5, color='firebrick')
    meanprops = dict(linewidth=1.5, color='#ff7f0e')
    # data = [data[:, i] for i in range(4)]
    bp1 = ax1.boxplot(
        datasf, 0, 'gD', showmeans=True, meanline=True,
        patch_artist=True, medianprops=medianprops,
        meanprops=meanprops)
    for patch, color in zip(bp1['boxes'], colordict.values()):
        patch.set_facecolor(color)
    # plt.xlim(0, len(mean))
    ax1.legend(zip(bp1['boxes']), labels, fontsize="small", loc="lower left", title='Distance')
    ax1.set_xticklabels(labels)
    ax1.set_yticks(range(0, 105, 20))
    ax1.set_xlabel('Site distance from hub')
    ax1.set_ylabel('Foraging (%)')
    # ax1.set_title('Swarm Foraging with distance '+ site[-2:])
    # ax2.set_title('Swarm Foraging with distance '+ site[-2:])
    plt.tight_layout()

    maindir = '/tmp/swarm/data/experiments/'
    # fname = 'agentsitecomp' + agent
    nadir = os.path.join(maindir, str(50), agent)

    fig.savefig(
        nadir + 'agentallsitecompdist' + '.png')
    # fig.savefig(
    #     maindir + '/' + fname + '.png')
    # pylint: disable = E1101

    plt.close(fig)


def boxplot_exp_0():
    agents = [50, 100, 200, 300, 400]
    dataf = [read_data_exp_3(100, 100, 5, 5, exp_no=0, site=30, no_trap=1, no_obs=1, agent=a)[0][:,-1] for a in agents]
    datad = [read_data_exp_3(100, 100, 5, 5, exp_no=0, site=30, no_trap=1, no_obs=1, agent=a)[1][:,-1] for a in agents]
    datadp = [(datad[d]/agents[d])*100.0 for d in range(len(datad))]
    fig = plt.figure(figsize=(6, 8), dpi=100)
    # dataall = [read_data_exp_3(100, 100, 5, 5, exp_no=0, site=30, no_trap=1, no_obs=1, agent=a)[0] for a in agents]
    # data = [data[:,-1] for data in dataall]

    ax1 = fig.add_subplot(3, 1, 1)
    colordict = {
        0: 'forestgreen',
        1: 'indianred',
        2: 'gold',
        3: 'tomato',
        4: 'royalblue'}
    colorshade = [
        'springgreen', 'lightcoral',
        'khaki', 'lightsalmon', 'deepskyblue']

    labels = [50, 100, 200, 300, 400]
    medianprops = dict(linewidth=1.5, color='firebrick')    # Strong line is median
    meanprops = dict(linewidth=2.5, color='#ff7f0e')    # Dashed line is mean
    bp1 = ax1.boxplot(
        dataf, 0, 'gD', showmeans=True, meanline=True,
        patch_artist=True, medianprops=medianprops,
        meanprops=meanprops)
    for patch, color in zip(bp1['boxes'], colordict.values()):
        patch.set_facecolor(color)
    ax1.legend(zip(bp1['boxes']), labels, fontsize="small", loc="lower right", title='No. of agents')
    ax1.set_xticklabels(labels)
    ax1.set_yticks(range(0, 105, 20))
    # ax1.set_xlabel('No. of agents ')
    ax1.set_ylabel('Foraging (%)', fontsize="large")
    ax1.set_title('Swarm Foraging', fontsize="large")

    ax2 = fig.add_subplot(3, 1, 2)
    bp2 = ax2.boxplot(
        datad, 0, 'gD', showmeans=True, meanline=True,
        patch_artist=True, medianprops=medianprops,
        meanprops=meanprops)
    for patch, color in zip(bp2['boxes'], colordict.values()):
        patch.set_facecolor(color)
    # plt.xlim(0, len(mean))
    # ax2.legend(zip(bp2['boxes']), labels, fontsize="small", loc="upper right", title='Agent Size')
    ax2.set_yticks([0, 100, 200, 300, 400])
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
    ax3.set_xticklabels(labels)
    ax3.set_xlabel('No. of agents', fontsize="large")
    ax3.set_ylabel('Dead Agents (%)', fontsize="large")
    ax3.set_yticks(range(0,105, 20))
    ax3.set_xticklabels(labels)

    # ax2 = fig.add_subplot(2, 1, 2)
    # for i in range(len(dataall)):
    #     medianf = np.quantile(dataall[i], 0.5, axis=0)
    #     q1f = np.quantile(dataall[i], 0.25, axis=0)
    #     q3f = np.quantile(dataall[i], 0.75, axis=0)
    #     xvalues = range(dataall[i].shape[1])
    #     ax2.plot(
    #         xvalues, medianf,
    #         linewidth=2.0, label=labels[i])
    #     # ax2.fill_between(
    #     #     xvalues, q3f, q1f,
    #     #     alpha=0.5)
    # ax2.legend(labels, fontsize="small", loc="lower right", title='No. of agents')
    # # ax1.set_xticklabels(labels)
    # ax2.set_yticks(range(0, 105, 20))
    # ax2.set_xlabel('Time Steps')
    # ax2.set_ylabel('Foraging (%)')
    # plt.tight_layout()

    maindir = '/tmp/swarm/data/experiments/'
    fname = 'agentscompboxplot'

    fig.savefig(
        maindir + '/' + fname + '.png')
    # pylint: disable = E1101

    plt.close(fig)


def experiment_1(agent='ExecutingAgent'):
    plt.style.use('fivethirtyeight')
    sites = [20, 25, 30, 40, 50]
    datasf = [read_data_exp_3(100, 100, 5, 5, exp_no=1, site=s, no_trap=1, no_obs=1)[0] for s in sites]
    # datasd = [read_data_exp_3(100, 100, 5, 5, exp_no=1, site=s, no_trap=1, no_obs=1)[1][:,-1] for s in sites]

    fig = plt.figure()

    ax1 = fig.add_subplot(1, 1, 1)
    colordict = {
        0: 'forestgreen',
        1: 'peru',
        2: 'royalblue',
        3: 'orchid',
        4: 'olivedrab',
        5: 'gold',
        6: 'linen',
        7: 'indianred',
        8: 'tomato'}
    colorshade = [
        'springgreen', 'lightcoral',
        'khaki', 'lightsalmon', 'deepskyblue', 'bisque']
    #     1: 'darkorange',
    #     2: 'orangered',
    #     3: 'seagreen']
    # colordict = {
    #     0: 'bisque',
    #     1: 'darkorange',
    #     2: 'orangered',
    #     3: 'seagreen'
    # }

    labels = [str(a) for a in sites]
    for i in range(len(datasf)):
        medianf = np.quantile(datasf[i], 0.5, axis=0)
        q1f = np.quantile(datasf[i], 0.25, axis=0)
        q3f = np.quantile(datasf[i], 0.75, axis=0)
        xvalues = range(datasf[i].shape[1])
        # print(len(xvalues), medianf.shape)
        ax1.plot(
            xvalues, medianf, '--',
            linewidth=2.0, label=labels[i])
        ax1.fill_between(
            xvalues, q3f, q1f,
            alpha=0.2)
    # plt.xlim(0, len(mean))
    # ax1.legend(zip(bp1['boxes']), labels, fontsize="small", loc="lower left", title='Distance')
    ax1.legend(labels, fontsize="small", loc="upper left", title='Distance')
    # ax1.set_xticklabels(labels)
    ax1.set_yticks(range(0, 105, 20))
    ax1.set_xlabel('Time Steps')
    ax1.set_ylabel('Foraging (%)')
    # ax1.set_title('Swarm Foraging with distance '+ site[-2:])
    # ax2.set_title('Swarm Foraging with distance '+ site[-2:])
    plt.tight_layout()

    maindir = '/tmp/swarm/data/experiments/'
    # fname = 'agentsitecomp' + agent
    nadir = os.path.join(maindir, agent)

    fig.savefig(
        nadir + 'agentallsitecompdistplot' + '.png')
    # fig.savefig(
    #     maindir + '/' + fname + '.png')
    # pylint: disable = E1101

    plt.close(fig)


def get_info_database(agent_size=50):
    ## To find the total no. of experiments from the data_base with agent_size=50
    # select count(*) from experiment where sn>=18169 and agent_size=50; -> 36
    # select count(*) from experiment where sn>=18169 and agent_size=50 and sucess=true; -> 34
    # Hit rate - 34/36*100 -> 94.44 %
    from swarms.utils.db import Connect, Execute

    connect = Connect('swarm', 'swarm', 'swarm', 'localhost')
    connect = connect.tns_connect()
    exestat = Execute(connect)
    total_exp = exestat.cursor.execute(
        "select count(*) from experiment where sn>=18169 and agent_size=" + "'" + str(agent_size) +
        "'")
    total_exp = exestat.cursor.fetchall()[0][0]
    total_sucess = exestat.cursor.execute(
        "select count(*) from experiment where sn>=18169 and sucess=true and agent_size=" + "'" + str(agent_size) +
        "'")
    total_sucess = exestat.cursor.fetchall()[0][0]
    hitrate = round((total_sucess / (total_exp * 1.0)) * 100, 2)

    time_value_data = exestat.cursor.execute(
        "select DATEDIFF('second',create_date::timestamp, end_date::timestamp) as runtime, total_value from experiment where sn>=18169 and sucess=true and agent_size=" + "'" + str(agent_size) +
        "'")
    time_value_data = exestat.cursor.fetchall()
    runtime, values = zip(*time_value_data)
    exestat.close()
    return [total_exp, total_sucess, hitrate, runtime, values]


def plot_evolution_algo_performance():
    plt.style.use('fivethirtyeight')
    agent_sizes = [50, 100, 150, 200]
    datas = [get_info_database(n) for n in agent_sizes]
    print('hitrates', agent_sizes, [data[2] for data in datas])
    runtime_data = [np.array(data[3]) for data in datas]
    values_data = [np.array(data[4]) for data in datas]
    fig = plt.figure()

    ax1 = fig.add_subplot(2, 1, 1)
    colordict = {
        0: 'forestgreen',
        1: 'gold',
        2: 'royalblue',
        3: 'orchid',
        4: 'olivedrab',
        5: 'peru',
        6: 'linen',
        7: 'indianred',
        8: 'tomato'}
    colorshade = [
        'springgreen', 'lightcoral',
        'khaki', 'lightsalmon', 'deepskyblue']

    labels = [str(a) for a in agent_sizes]
    medianprops = dict(linewidth=2.5, color='firebrick')
    meanprops = dict(linewidth=2.5, color='#ff7f0e')
    # data = [data[:, i] for i in range(4)]
    bp1 = ax1.boxplot(
        runtime_data, 0, 'gD', showmeans=True, meanline=True,
        patch_artist=True, medianprops=medianprops,
        meanprops=meanprops)
    for patch, color in zip(bp1['boxes'], colordict.values()):
        patch.set_facecolor(color)
    # plt.xlim(0, len(mean))
    ax1.legend(zip(bp1['boxes']), labels, fontsize="small", loc="lower right", title='Agent Size')
    ax1.set_xticklabels(labels)
    ax1.set_xlabel('Agent size')
    ax1.set_ylabel('Run Time (Secs)')

    ax2 = fig.add_subplot(2, 1, 2)
    bp2 = ax2.boxplot(
        values_data, 0, 'gD', showmeans=True, meanline=True,
        patch_artist=True, medianprops=medianprops,
        meanprops=meanprops)
    for patch, color in zip(bp2['boxes'], colordict.values()):
        patch.set_facecolor(color)
    ax1.legend(zip(bp1['boxes']), labels, fontsize="small", loc="lower right", title='Agent Size')
    ax2.set_xticklabels(labels)
    ax2.set_xlabel('Agent size')
    ax2.set_ylabel('Foraging (%)')

    plt.tight_layout()
    maindir = '/tmp/swarm/data/experiments/'
    # fname = 'agentsitecomp' + agent
    nadir = os.path.join(maindir, str(50))

    fig.savefig(
        nadir + 'evolutionperform' + '.png')
    plt.close(fig)


def read_data_sample_ratio(ratio=0.1):
    # maindir = '/tmp/swarm/data/experiments/behavior_sampling'
    # maindir = '/tmp/swarms/data/experiment/'
    ## Experiment ID for the plots/results in the paper
    # maindir = '/tmp/16244729911974EvoSForgeNewPPA1/'
    # maindir = '/tmp/experiments/100/12000/16243666378807EvoSForgeNewPPA1'
    # nadir = os.path.join(maindir, str(n), agent)
    maindir = '/tmp/16244729911974EvoSForgeNewPPA1/'  # New sampling behaviors
    folders = pathlib.Path(maindir).glob("*_" + str(ratio) + "_ValidateSForgeNewPPA1")
    flist = []
    data = []
    for f in folders:
        try:
            flist = [p for p in pathlib.Path(f).iterdir() if p.is_file() and p.match('simulation.csv')]
            _, _, d = np.genfromtxt(flist[0], autostrip=True, unpack=True, delimiter='|')
            data.append(d)
        except:
            pass
    data = np.array(data)
    # print(ratio, data.shape)
    return data


def plot_sampling_differences():
    plt.style.use('fivethirtyeight')
    sampling_size = [0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
    datas = []
    for s in sampling_size:
        data = read_data_sample_ratio(s)[:, -1]
        print(s, data.shape)
        datas.append(data)

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
    labels = sampling_size
    medianprops = dict(linewidth=2.5, color='firebrick')
    meanprops = dict(linewidth=2.5, color='#ff7f0e')
    bp1 = ax1.boxplot(
        datas, 0, 'gD', showmeans=True, meanline=True,
        patch_artist=True, medianprops=medianprops,
        meanprops=meanprops)
    for patch, color in zip(bp1['boxes'], colordict.values()):
        patch.set_facecolor(color)
    # plt.xlim(0, len(mean))
    ax1.legend(zip(bp1['boxes']), labels, fontsize="small", loc="lower left", title='Sampling Size')
    ax1.set_xticklabels(labels)
    ax1.set_yticks(range(0, 105, 20))
    ax1.set_xlabel('Sampling Size')
    ax1.set_ylabel('Foraging %')
    ax1.set_title('Swarm Foraging')

    plt.tight_layout()

    maindir = '/tmp/swarm/data/experiments/'
    fname = 'behavior_sampling'

    fig.savefig(
        maindir + '/' + fname + '.png')

    plt.close(fig)


def read_data_n(n=100, comm=True):
    maindir = '/tmp/swarm/data/experiments/'
    if comm:
        folders = pathlib.Path(maindir + 'comm').glob('*SForgeNewPPAComm1')
    else:
        folders = pathlib.Path(maindir + 'withoutcomm').glob('*ForagingSimulation')
    flist = []
    dataf = []
    datad = []
    for f in folders:
        try:
            flist = [p for p in pathlib.Path(f).iterdir() if p.is_file() and p.match('simulation.csv')]
            # print(flist)
            _, _, f, d = np.genfromtxt(flist[0], autostrip=True, unpack=True, delimiter='|')
            dataf.append(f)
            datad.append(d)
        except:
            pass
    dataf = np.array(dataf)
    datad = np.array(datad)
    return dataf, datad


def read_data_exp_3(width=100, height=100, trap=5, obs=5, exp_no=3, site=30, no_trap=1, no_obs=1, agent=100, grid=None):
    maindir = '/tmp/swarm/data/experiments/'
    if grid is None:
        ndir = os.path.join(maindir, str(agent), 'ExecutingAgent', str(exp_no),
                str(site), str(trap)+'_'+str(obs), str(no_trap)+'_'+str(no_obs), str(width) +'_'+str(height))
    else:
        ndir = os.path.join(maindir, str(agent), 'ExecutingAgent', str(exp_no),
                str(site), str(trap)+'_'+str(obs), str(no_trap)+'_'+str(no_obs), str(width) +'_'+str(height), str(grid))
    print(ndir)
    folders = pathlib.Path(ndir).glob('*ForagingSimulation')
    flist = []
    dataf = []
    datad = []
    for f in folders:
        try:
            flist = [p for p in pathlib.Path(f).iterdir() if p.is_file() and p.match('simulation.csv')]
            # print(flist)
            _, _, f, d = np.genfromtxt(flist[0], autostrip=True, unpack=True, delimiter='|')
            dataf.append(f)
            datad.append(d)
        except:
            pass
    dataf = np.array(dataf)
    datad = np.array(datad)
    return dataf, datad


def boxplot_exp_3():
    size = [100, 200, 300, 400, 500, 600]
    data = [read_data_exp_3(s, s, 5, 5, exp_no=3, site=30, no_trap=1, no_obs=1)[0][:,-1] for s in size]
    fig = plt.figure()
    ax1 = fig.add_subplot(1, 1, 1)
    colordict = {
        0: 'forestgreen',
        1: 'indianred',
        2: 'gold',
        3: 'tomato',
        4: 'royalblue',
        5: 'peru'}

    labels = [100, 200, 300, 400, 500, 600]
    labels = [str(l) +'x'+ str(l) for l in  labels]
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
    ax1.legend(zip(bp1['boxes']), labels, fontsize="large", loc="upper right", title='Environment Size')
    ax1.set_xticklabels(labels, fontsize='large')
    ax1.set_xlabel('Environment Size [Width x Height]', fontsize='large')
    ax1.set_ylabel('Foraging (%)', fontsize='large')
    ax1.set_yticks(range(0, 105, 20))
    # ax1.set_title('Swarm Foraging Evolved Behaviors')

    plt.tight_layout()

    maindir = '/tmp/swarm/data/experiments/'
    fname = 'environment_size'

    fig.savefig(
        maindir + '/' + fname + '.png')
    # pylint: disable = E1101

    plt.close(fig)


def boxplot_exp_2():
    size = [5, 10, 15, 20, 25]
    dataf = [read_data_exp_3(100, 100, s, s, exp_no=2, site=30, no_trap=1, no_obs=1)[0][:,-1] for s in size]
    datad = [read_data_exp_3(100, 100, s, s, exp_no=2, site=30, no_trap=1, no_obs=1)[1][:,-1] for s in size]
    # dataf = [read_data_exp_3(100, 100, s, s, exp_no=2)[0][:,-1] for s in size]
    # datad = [read_data_exp_3(100, 100, s, s, exp_no=2)[1][:,-1] for s in size]
    datadp = [(d/100)*100.0 for d in datad]
    fig = plt.figure(figsize=(6, 8), dpi=100)

    ax1 = fig.add_subplot(3, 1, 1)
    colordict = {
        0: 'forestgreen',
        1: 'indianred',
        2: 'gold',
        3: 'tomato',
        4: 'royalblue',
        5: 'peru'}

    labels = [5, 10, 15, 20, 25]
    labels = [str(l) for l in  labels]
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
    ax1.legend(zip(bp1['boxes']), labels, fontsize="small", loc="lower left", title='Trap/Obstacle Size')
    ax1.set_xticklabels(labels)
    # ax1.set_xlabel('Trap/Obstacle Size', fontsize='large')
    ax1.set_ylabel('Foraging (%)',  fontsize='large')
    ax1.set_yticks(range(0, 105, 20))
    # ax1.set_title('Swarm Foraging Evolved Behaviors')
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
    ax3.set_xlabel('Trap/Obstacle Size', fontsize="large")
    ax3.set_ylabel('Dead Agents (%)', fontsize="large")
    plt.tight_layout()

    maindir = '/tmp/swarm/data/experiments/'
    fname = 'trap_obstacle_size'

    fig.savefig(
        maindir + '/' + fname + '.png')
    # pylint: disable = E1101

    plt.close(fig)


def boxplot_exp_4():
    size = [1, 2, 3, 4, 5]
    dataf = [read_data_exp_3(100, 100, 5, 5, exp_no=4, site=30, no_trap=s, no_obs=s)[0][:,-1] for s in size]
    datad = [read_data_exp_3(100, 100, 5, 5, exp_no=4, site=30, no_trap=s, no_obs=s)[1][:,-1] for s in size]
    datadp = [(d/100)*100.0 for d in datad]
    fig = plt.figure(figsize=(6, 8), dpi=100)

    ax1 = fig.add_subplot(3, 1, 1)
    colordict = {
        0: 'forestgreen',
        1: 'indianred',
        2: 'gold',
        3: 'tomato',
        4: 'royalblue',
        5: 'peru'}

    labels = [5, 10, 15, 20, 25]
    labels = [str(l) for l in  labels]
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
    ax1.set_xticklabels(labels)
    # ax1.set_xlabel('Trap/Obstacle No.')
    ax1.set_ylabel('Foraging (%)', fontsize='large')
    ax1.set_yticks(range(0, 105, 20))
    # ax1.set_title('Swarm Foraging Evolved Behaviors')
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
    ax3.legend(zip(bp1['boxes']), labels, fontsize="small", loc="lower right", title='Trap/Obstacle No.')
    ax3.set_yticks(range(0,105, 20))
    ax3.set_xticklabels(labels)
    ax3.set_xlabel('Trap/Obstacle No.', fontsize="large")
    ax3.set_ylabel('Dead Agents (%)', fontsize="large")
    plt.tight_layout()

    maindir = '/tmp/swarm/data/experiments/'
    fname = 'trap_obstacle_no'

    fig.savefig(
        maindir + '/' + fname + '.png')
    # pylint: disable = E1101

    plt.close(fig)


def boxplot_exp_5(t=1):
    grids = [2, 5, 10]
    dataf = [read_data_exp_3(100, 100, t, t, exp_no=5, site=30, no_trap=1, no_obs=1, grid=g)[0][:,-1] for g in grids]
    datad = [read_data_exp_3(100, 100, t, t, exp_no=5, site=30, no_trap=1, no_obs=1, grid=g)[1][:,-1] for g in grids]
    datadp = [(d/100)*100.0 for d in datad]
    fig = plt.figure(figsize=(6, 8), dpi=100)

    ax1 = fig.add_subplot(3, 1, 1)
    colordict = {
        0: 'forestgreen',
        1: 'indianred',
        2: 'gold',
        3: 'tomato',
        4: 'royalblue',
        5: 'peru'}

    labels = [2, 5, 10]
    labels = [str(l) for l in  labels]
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
    ax1.legend(zip(bp1['boxes']), labels, fontsize="small", loc="lower left", title='Grid Size')
    ax1.set_xticklabels(labels)
    # ax1.set_xlabel('Trap/Obstacle Size', fontsize='large')
    ax1.set_ylabel('Foraging (%)',  fontsize='large')
    ax1.set_yticks(range(0, 105, 20))
    # ax1.set_title('Swarm Foraging Evolved Behaviors')
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
    ax3.set_xlabel('Grid Size', fontsize="large")
    ax3.set_ylabel('Dead Agents (%)', fontsize="large")
    plt.tight_layout()

    maindir = '/tmp/swarm/data/experiments/'
    fname = 'grid_size'

    fig.savefig(
        maindir + '/' + fname + '_' + str(t) +'.png')
    # pylint: disable = E1101

    plt.close(fig)


def comp_with_witout_comm():
    datac, datacd = read_data_n(n=100, comm=True)
    datawc, datawcd = read_data_n(n=100, comm=False)
    print(datac.shape, datawc.shape)

    def return_quartiles(datas):
        median = np.quantile(datas, 0.5, axis=0)
        q1 = np.quantile(datas, 0.25, axis=0)
        q3 = np.quantile(datas, 0.75, axis=0)
        return median, q1, q3

    medianc, q1c, q3c = return_quartiles(datac)
    medianwc, q1wc, q3wc = return_quartiles(datawc)

    color = [
        'forestgreen', 'indianred',
        'gold', 'tomato', 'royalblue']
    colorshade = [
        'springgreen', 'lightcoral',
        'khaki', 'lightsalmon', 'deepskyblue']

    plt.style.use('fivethirtyeight')
    fig = plt.figure()
    ax1 = fig.add_subplot(2, 1, 1)
    xvalues = range(datac.shape[1])
    ax1.plot(
        xvalues, medianc, color=color[0],
        linewidth=1.0, label='With Communication')
    ax1.fill_between(
        xvalues, q3c, q1c,
        color=colorshade[0], alpha=0.5)

    ax1.plot(
        xvalues, medianwc, color=color[1],
        linewidth=1.0, label='Without Communication')
    ax1.fill_between(
        xvalues, q3wc, q1wc,
        color=colorshade[1], alpha=0.5)

    ax1.set_xlabel('Steps')
    ax1.set_ylabel('Foraging %')
    ax1.set_yticks(range(0, 105, 20))
    ax1.legend()
    mediancd, q1cd, q3cd = return_quartiles(datacd)
    medianwcd, q1wcd, q3wcd = return_quartiles(datawcd)

    ax2 = fig.add_subplot(2, 1, 2)
    xvalues = range(datac.shape[1])
    ax2.plot(
        xvalues, mediancd, color=color[0],
        linewidth=1.0, label='With Communication')
    ax2.fill_between(
        xvalues, q3cd, q1cd,
        color=colorshade[0], alpha=0.5)

    ax2.plot(
        xvalues, medianwcd, color=color[1],
        linewidth=1.0, label='Without Communication')
    ax2.fill_between(
        xvalues, q3wcd, q1wcd,
        color=colorshade[1], alpha=0.5)

    ax2.set_xlabel('Steps')
    ax2.set_ylabel('No. Dead Agents')
    ax2.set_yticks(range(0, 20, 4))
    # ax1.set_xticks(
    #     np.linspace(0, data.shape[-1], 5))
    # plt.legend()
    plt.tight_layout()

    # fig.savefig(
    #     '/tmp/goal/data/experiments/' + pname + '.pdf')
    maindir = '/tmp/swarm/data/experiments/'
    # nadir = os.path.join(maindir, str(n), agent)
    fig.savefig(
        maindir + 'comparewithoutcomm.png')
    plt.close(fig)


def main():
    # read_data_n_agent()
    # plotgraph()
    # boxplot()

    # sitelocation  = [
    #     {"x":51, "y":-51, "radius":10, "q_value":0.9},
    #     {"x":51, "y":51, "radius":10, "q_value":0.9},
    #     {"x":-51, "y":51, "radius":10, "q_value":0.9},
    #     {"x":31, "y":-31, "radius":10, "q_value":0.9},
    #     {"x":31, "y":31, "radius":10, "q_value":0.9},
    #     {"x":-31, "y":31, "radius":10, "q_value":0.9},
    #     {"x":91, "y":-91, "radius":10, "q_value":0.9},
    #     {"x":-91, "y":91, "radius":10, "q_value":0.9},
    # ]
    # for site in sitelocation:
    #     for n in [50, 100, 200, 300, 400]:
    #         # plotgraphsite(n=n, agent='ExecutingAgent', site=str(site['x'])+str(site['y']))
    #         boxplotsiteloc(site=str(site['x'])+str(site['y']))

    # boxplotallsites()
    # boxplotagent()
    # boxplotallsitesdist()
    # boxplotsiteloc()
    # plot_evolution_algo_performance()
    # plot_sampling_differences()
    # plotallsitesdist()
    # comp_with_witout_comm()
    # boxplot_exp_0()
    # experiment_1()
    # boxplot_exp_2()
    # boxplot_exp_3()
    # boxplot_exp_4()
    for t in [1, 3, 5, 9, 15]:
        boxplot_exp_5(t)


if __name__ == '__main__':
    main()