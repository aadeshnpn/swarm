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


def load_grammar(gname='/tmp/coevo.bnf'):
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
        'terminals: %i, non-terminals: %i, mcc: %i, arhs: %0.2f, hal: %0.2f ' %
        (T, N, compute_MCC(gname), average_RHS(gname, N), compute_HAL(gname, T, N)))

    # print(compute_HAL(gname, T, N))
    # print(dir(bnf))
    # print(compute_MCC(gname))
    # print(average_RHS(gname, len(bnf.non_terminals)))


def main():
    load_grammar()


if __name__ == '__main__':
    main()
