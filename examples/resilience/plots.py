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
    median = np.quantile(dataf, 0.5, axis=0)
    # print(median)
    q1 = np.quantile(dataf, 0.25, axis=0)
    q3 = np.quantile(dataf, 0.75, axis=0)
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
        xvalues, median, color=color[0],
        linewidth=1.0)
    ax1.fill_between(
        xvalues, q3, q1,
        color=colorshade[0], alpha=0.3)
    plt.title('Foraging')
    # ax1.legend(title='$\it{m}$')
    ax1.set_xlabel('Steps')
    ax1.set_ylabel('Foraging %')

    # ax1.set_xticks(
    #     np.linspace(0, data.shape[-1], 5))
    plt.tight_layout()

    # fig.savefig(
    #     '/tmp/goal/data/experiments/' + pname + '.pdf')
    maindir = '/tmp/swarm/data/experiments/'
    nadir = os.path.join(maindir, str(n), agent)    
    fig.savefig(
        nadir + 'foraging' + '.png')
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



def boxplotsiteloc(agent='SimForgAgentWith'):
    # sites = ['-3030', '30-30', '3030', '-5050', '50-50', '5050', '-9090', '90-90']
    agents = [100, 200, 300, 400]
    data = [read_data_n_agent_site(n, agent, site='50-50')[:,-1] for n in agents]
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
    # labels = ['30', '30', '30', '50', '50', '50', '90', '90']
    labels = ['100', '200', '300', '400']
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
    ax1.legend(zip(bp1['boxes']), labels, fontsize="small", loc="upper right", title='Agent Size')
    ax1.set_xticklabels(labels)
    ax1.set_xlabel('Agent size')
    ax1.set_ylabel('Foraging Percentage')
    ax1.set_title('Swarm Foraging with distance 50')

    plt.tight_layout()

    maindir = '/tmp/swarm/data/experiments/'
    fname = 'agentsitecomp' + agent

    fig.savefig(
        maindir + '/' + fname + '.png')
    # pylint: disable = E1101

    plt.close(fig)


def main():
    # agents = [50, 100, 200, 300, 400]
    atype = ['SimForgAgentWith', 'SimForgAgentWithout']
    # boxplotsiteloc(atype[1])
    # boxplot(atype[1])    
    sitelocation  = [ 
        {"x":50, "y":-50, "radius":10, "q_value":0.9},
        {"x":50, "y":50, "radius":10, "q_value":0.9},        
        {"x":-50, "y":50, "radius":10, "q_value":0.9},                
        {"x":30, "y":-30, "radius":10, "q_value":0.9},
        {"x":30, "y":30, "radius":10, "q_value":0.9},        
        {"x":-30, "y":30, "radius":10, "q_value":0.9},                        
        {"x":90, "y":-90, "radius":10, "q_value":0.9},
        {"x":-90, "y":90, "radius":10, "q_value":0.9}, 
    ]    
    i = 2
    sitename = str(sitelocation[i]['x']) + str(sitelocation[i]['y'])
    print(sitename)
    plotgraph(n=100, agent=atype[1], site=sitename)    
    plotgraph(n=100, agent=atype[0], site=sitename)        
    # agents = [100, 200, 300, 400]
    # for n in agents:
    #     plotgraph(n=n, agent=atype[1])
    # for n in agents:
    #     for t in atype:
    #         plotgraph(n=n, agent=t)
    # for t in atype:
    #     boxplot(t)


# import os
# dname = os.path.join('/tmp', 'pygoal', 'data', 'experiments')
# pathlib.Path(dname).mkdir(parents=True, exist_ok=True)
# fname = os.path.join(dname, name + '.png')
# fig.savefig(fname)
# plt.close(fig)


if __name__ == '__main__':
    main()