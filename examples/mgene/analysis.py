from swarms.utils.bt import BTConstruct
from swarms.utils.jsonhandler import JsonPhenotypeData
import pathlib
from swarms.lib.model import Model
from swarms.lib.agent import Agent
from swarms.behaviors.scbehaviors import (
    CompositeDrop, CompositeSingleCarry, MoveTowardsNormal, MoveAwayNormal, ExploreNormal)
import numpy as np


def read_phenotypes():
    # maindir = '/tmp/swarm/data/experiments/seqseq/CombinedModelPPA/100/12000/10/200/2/None/None/1/5/10000/0.85/10/'
    maindir = '/tmp/combined_old/'
    folders = pathlib.Path(maindir).glob("166*" + "CombinedModelPPA")
    flist = []
    data = []
    fitness = []
    for f in folders:
        flist = [p for p in pathlib.Path(f).iterdir() if p.is_file() and (p.match('*_all.json') or p.match('simulation.csv') )]
        print(flist)
        if len(flist) ==2:
            xmlstringsall = JsonPhenotypeData.load_json_file(flist[1])
            xmlstringsall = xmlstringsall['phenotypes']
            data.append(xmlstringsall)
            _, _, d = np.genfromtxt(flist[0], autostrip=True, unpack=True, delimiter='|')
            # print(flist, d[-1])
            fitness.append(d[-1])

    return data, fitness


def compute_complete(xmlstring):
    model = Model()
    agent = Agent(name=1, model=model)
    bt = BTConstruct(None, agent)
    bt.xmlstring = xmlstring
    bt.construct()
    allnodes = list(bt.behaviour_tree.root.iterate())
    # print(allnodes)
    carry = list(filter(
        lambda x: isinstance(x, CompositeSingleCarry), allnodes)
        )
    drop = list(filter(
        lambda x: isinstance(x, CompositeDrop), allnodes)
        )
    away = list(filter(
        lambda x: isinstance(x, MoveAwayNormal), allnodes)
        )
    towards = list(filter(
        lambda x: isinstance(x, MoveTowardsNormal), allnodes)
        )
    explore = list(filter(
        lambda x: isinstance(x, ExploreNormal), allnodes)
        )
    if (
        len(carry) == 1 and len(drop) == 1 and len(
            away) ==1 and len(towards) ==1 and len(
                explore) ==1):
        return True
    else:
        return False


def compute_phenotype_properties():
    phenotypes,fitness = read_phenotypes()
    i=0
    complete_phenotype = []
    for simulation in phenotypes:
        complete = []
        for agent in simulation:
            complete.append(compute_complete(agent))
        print('Complete Actions: ', sum(complete), 'Final Fitness: ', fitness[i])
        # break
        complete_phenotype.append(sum(complete))
        i +=1
    print(complete_phenotype)
    np.save('/tmp/seq2seq_sequence.npy', complete_phenotype)


def main():
    compute_phenotype_properties()
    # [0, 48, 48, 4, 4, 2, 20, 3, 0, 35, 2, 1, 45, 3, 0, 1, 4, 31, 4, 4, 3]
    # [1, 3, 3, 1, 0, 2, 0, 0, 0, 0, 0, 6, 0, 2, 2, 0, 1, 4, 2, 1, 9, 2, 0, 5, 0, 2, 9, 0, 3, 1, 1, 2, 1, 12, 0, 1, 1, 2, 4, 0, 3, 0, 0, 0, 22]


if __name__ == '__main__':
    main()