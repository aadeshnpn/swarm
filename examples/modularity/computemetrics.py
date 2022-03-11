from ponyge.representation.grammar import Grammar


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


def average_RHS(fname):
    pass


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
    print(
        'terminals: %i, non-terminals: %i, mcc: %i ' % (
        len(bnf.terminals), len(bnf.non_terminals), compute_MCC(gname)))

    # print(dir(bnf))
    # print(compute_MCC(gname))


def main():
    load_grammar()


if __name__ == '__main__':
    main()
