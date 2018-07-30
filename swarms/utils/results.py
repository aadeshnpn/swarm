# Store the information from the experiments

import datetime


class Results:
    """Define the results atrributes.

    This class defines the common attributes for each experiments.
    """
    def __init__(
        self, foldername, agent_name, step, beta, fitness, diversity, explore,
        foraging, neighbour, genotype, phenotype, bt
            ):
        """Initialize the attributes."""
        self.foldername = foldername
        self.context = {
            "id": datetime.datetime.now().strftime("%s"),
            "name": agent_name,
            "step": step,
            "beta": beta,
            "fitness": fitness,
            "diversity": diversity,
            "explore": explore,
            "foraging": foraging,
            "neighbour": neighbour,
            "genotype": genotype,
            "phenotype": phenotype,
            "bt_source": bt
        }
        # self.template = """
        # Id, Agent Name, Step, Beta, Fitness, Diversity, Explore, Foraging, Neighbours, Genotype, Phenotype, BT
        self.template = """
        {id}, {name}, {step}, {beta}, {fitness}, {diversity}, {explore}, {foraging}, {neighbour}, {genotype}, {phenotype}, {bt}
        """

    def save_to_file(self):
        """Save results to a flat file."""
        filename = self.foldername + '/' + str(self.context['step']) + '_' + str(self.context['name'])
        with open(filename, 'a') as statsfile:
            statsfile.write(self.template.format(**self.context))

    def save_to_db(self):
        """Save results to a database."""
        pass
