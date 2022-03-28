from ponyge.representation.grammar import Grammar
import re
import math
import copy


def compute_MCC(fname):
    # For each production rule
    mcc = 0
    with open(fname, 'r') as f:
        productions = f.readlines()
    for production in productions:
        # production = f.readline()
        if production.rstrip():
            # print(production.split('|'))
            rhsdecision = len(production.split('|'))
            if rhsdecision > 1:
                mcc += rhsdecision
    return mcc


def compute_shis(fname):
    # For each production rule
    shi1 = 0
    shi2 = 0
    with open(fname, 'r') as f:
        productions = f.readlines()
    for production in productions:
        # production = f.readline()
        if production.rstrip():
            # print(production.split('|'))s
            shi1 += 1 + len(re.findall(r'\|', production))
            rhsdecision = len(production.split('|'))
            shi2 += rhsdecision
    return shi1, shi2


def average_RHS(fname, nt):
    sizes = 0
    with open(fname, 'r') as f:
        productions = f.readlines()
    for i, production in enumerate(productions):
        if production.rstrip():
            rhs = production.split('::=')[1]
            rhsm = rhs.replace('<', '(')
            rhsm = rhsm.replace('>', ')')
            sizes += len(re.findall(r'\(((?:\w+\s*)+)\)', rhsm))

            all = rhs.split('|')
            for a in all:
                b = a.split('_')[0]
                if b.isalpha():
                    sizes += 1

    return round(sizes/nt, 2)


def compute_HAL(fname, T, N):
    meu1 = 1
    meu2 = T + N
    shi1, shi2 = compute_shis(fname)
    hal = (meu1 * shi2 * (shi1 + shi2) * math.log2(meu1 + meu2)) / (2 * meu2)
    return round(hal, 2)


def size_metrics(gname='/tmp/coevo.bnf'):
    # Grammatical Evolution part
    from ponyge.algorithm.parameters import Parameters
    parameter = Parameters()
    parameter_list = ['--parameters', '../..,modular.txt']
    # Comment when different results is desired.
    # Else set this for testing purpose
    # parameter.params['RANDOM_SEED'] = name
    # # np.random.randint(1, 99999999)
    # Set GE runtime parameters
    parameter.params['POPULATION_SIZE'] = 10
    parameter.set_params(parameter_list)
    bnf = Grammar(parameter, gname)
    T = len(set(list(bnf.terminals.keys())))
    N = len(set(list(bnf.non_terminals.keys())))
    print(
        gname + ', terminals: %i, non-terminals: %i, mcc: %i, arhs: %0.2f, hal: %0.2f ' %
        (T, N, compute_MCC(gname), average_RHS(gname, N), compute_HAL(gname, T, N)))

    # print(compute_HAL(gname, T, N))
    # print(dir(bnf))
    # print(compute_MCC(gname))
    # print(average_RHS(gname, len(bnf.non_terminals)))


def structure_metrics(gname='/tmp/coevo.bnf'):
    from ponyge.algorithm.parameters import Parameters
    parameter = Parameters()
    parameter_list = ['--parameters', '../..,modular.txt']
    parameter.params['POPULATION_SIZE'] = 10
    parameter.set_params(parameter_list)
    bnf = Grammar(parameter, gname)
    all_non_terminals = bnf.non_terminals.keys()
    immsuccessors = dict()
    for non_term_key in all_non_terminals:
        # print(non_term_key + '::= ', end=" ")
        immsuccessors[non_term_key] = []
        if len(bnf.non_terminals[non_term_key]['recursive']) > 0:
            for j in range(len(bnf.non_terminals[non_term_key]['recursive'])):
                for choice in bnf.non_terminals[non_term_key]['recursive'][j]['choice']:
                    if choice['type'] == 'NT':
                        # print(choice['symbol'], end=' ')
                        immsuccessors[non_term_key].append(choice['symbol'])
        # print()

    succesor = dict()
    for non_term_key in all_non_terminals:
        succesor[non_term_key] = []
        immsucc = immsuccessors[non_term_key]
        for i in immsucc:
            succesor[non_term_key] += immsuccessors[i]

    equivalence = dict()
    for k, val in succesor.items():
        for v in val:
            if k in succesor[v]:
                equivalence[k] = v

    timp = compute_TIMP(immsuccessors)
    clev = compute_CLEV(len(immsuccessors.keys()), len(equivalence.keys()))
    nslev = compute_NSLEV(equivalence, succesor)
    dep = compute_DEP(equivalence, succesor)
    print(
        gname + ', TIMP: %0.2f, CLEV: %0.2f, NSLEV: %0.2f, DEP: %0.2f' % (
            timp, clev, nslev, dep))


def compute_TIMP(immsucc):
    n = len(immsucc.keys())
    e = sum([len(v) for _, v in immsucc.items()])
    return (2 * (e - n + 1) * 100) / ((n-1) * (n-2))


def compute_CLEV(Neq, N):
    return (Neq * 100) / N


def compute_NSLEV(equivalance, succesor):
    count = 0
    for k in equivalance.keys():
        if len(set(succesor[k])) > 1:
            count += 1
    return count


def compute_DEP(equivalance, succesor):
    count = 0
    for k in equivalance.keys():
        if len(set(succesor[k])) > count:
            count = len(set(succesor[k]))
    return count


def main():
    gnames = [
        '/tmp/coevo.bnf', '/tmp/bt.bnf', '/tmp/swarm.bnf',
        '/tmp/coevoAddition.bnf']
    for gname in gnames:
        size_metrics(gname)
        structure_metrics(gname)


if __name__ == '__main__':
    main()
