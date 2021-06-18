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


def read_data_n_agent_site(n=100, agent='ExecutingAgent', site='5151'):
    # maindir = '/tmp/swarm/data/experiments/'
    maindir = '/home/aadeshnpn/Desktop/evolved_ppa/experiments/'
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
    print(dataf.shape, datad.shape, n)
    return dataf, datad


def boxplotagent(agent='ExecutingAgent', site='51-51'):
    # data = [read_data_n_agent(n, agent)[:,-1] for n in [50, 100, 200, 300, 400]]
    # data = [read_data_n_agent(n, agent)[:,-1] for n in [50, 100]]
    # data = [read_data_n_agent_site(n, agent, site=site)[0][:,-1] for n in agents]
    agents = [50, 100, 200, 300, 400]
    # print(agent, site)
    data = [read_data_n_agent_site(n, agent, site=site)[0][:,-1] for n in agents]
    # datad = [read_data_n_agent_site(n, agent, site=site)[1][:,-1] for n in agents]
    # datadp = [(datad[i]/agents[i])*100 for i in range(len(agents))]
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
    labels = [50, 100, 200, 300, 400]
    # labels = [50, 100]
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



def boxplotsiteloc(agent='ExecutingAgent', site='5151'):
    # sites = ['-3030', '30-30', '3030', '-5050', '50-50', '5050', '-9090', '90-90']
    agents = [50, 100, 200, 300, 400]
    # agents = [100]
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
    ax3.set_ylabel('Dead Agents %')
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
    sites = ['-3131', '31-31', '3131', '-5151', '51-51', '5151', '-9191', '91-91']
    # sites = ['-3131', '31-31', '3131', '-5151', '5151', '-9191', '91-91']
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
    labels = ['30', '30', '30', '50', '50', '50', '90', '90']
    # labels = ['50', '100'] #, '200', '300', '400']
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
    ax1.legend(zip(bp1['boxes']), labels, fontsize="small", loc="upper right", title='Distance')
    ax1.set_xticklabels(labels)
    ax1.set_xlabel('Site distance from hub')
    ax1.set_ylabel('Foraging Percentage')
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
    plot_evolution_algo_performance()


if __name__ == '__main__':
    main()