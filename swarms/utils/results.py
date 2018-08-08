"""Store the information from the experiments."""

import datetime
from pathlib import Path


class Results:
    """Define the results atrributes.

    This class defines the common attributes for each experiments for each
    agents.
    """

    def __init__(
        self, foldername, agent_name, step, timestep, beta, fitness, diversity,
        explore, foraging, neighbour, genotype, phenotype, bt
            ):
        """Initialize the attributes."""
        self.foldername = foldername
        self.context = {
            "id": datetime.datetime.now().strftime("%s"),
            "name": agent_name,
            "step": step,
            "timestep": timestep,
            "beta": beta,
            "fitness": fitness,
            "diversity": diversity,
            "explore": explore,
            "foraging": foraging,
            "neighbour": neighbour,
            "genotype": genotype,
            "phenotype": phenotype,
            "bt": bt
        }
        # self.template = """
        # Id, Agent Name, Step, Beta, Fitness, Diversity, Explore, Foraging,\
        # Neighbours, Genotype, Phenotype, BT
        self.template = """{id}|{name}|{step}|{timestep}|{beta}|{fitness}|{diversity}|{explore}|{foraging}|{neighbour}|{genotype}|{phenotype}|{bt}
        """
        # Write a header to the file for pandas dataframe
        self.header = """id|name|step|timestep|beta|fitness|diversity|explore|foraging|neighbour|genotype|phenotype|bt\n
        """

    def save_to_file(self):
        """Save results to a flat file."""
        filename = self.foldername + '/' + str(
            self.context['step']) + '.csv'  # + '_' + str(self.context['name'])
        # Create a path to the filename
        result_file = Path(filename)
        # Check if the file exists
        if result_file.is_file():
            with open(filename, 'a') as statsfile:
                statsfile.write(self.template.format(**self.context))
        else:
            with open(filename, 'a') as statsfile:
                statsfile.write(self.header)
                statsfile.write(self.template.format(**self.context))

    def save_to_db(self):
        """Save results to a database."""
        pass


class Best:
    """Define the results atrributes.

    This class defines the best attributes for each experiments for all
    agents.
    """

    def __init__(
        self, foldername, agent_name, header, step, beta, fitness, diversity,
        explore, foraging, phenotype
            ):
        """Initialize the attributes."""
        self.foldername = foldername
        self.context = {
            "id": datetime.datetime.now().strftime("%s"),
            "name": agent_name,
            "header": header,
            "step": step,
            "beta": beta,
            "fitness": fitness,
            "diversity": diversity,
            "explore": explore,
            "foraging": foraging,
            "phenotype": phenotype
        }

        self.template = """{id}|{header}|{name}|{step}|{beta}|{fitness}|{diversity}|{explore}|{foraging}|{phenotype}
        """
        # Write a header to the file for pandas dataframe
        self.header = """id|header|name|step|beta|fitness|diversity|explore|foraging|phenotype\n
        """

    def save_to_file(self):
        """Save results to a flat file."""
        filename = self.foldername + '/' + 'best.csv'
        # Create a path to the filename
        result_file = Path(filename)
        # Check if the file exists
        if result_file.is_file():
            with open(filename, 'a') as statsfile:
                statsfile.write(self.template.format(**self.context))
        else:
            with open(filename, 'a') as statsfile:
                statsfile.write(self.header)
                statsfile.write(self.template.format(**self.context))

    def save_to_db(self):
        """Save results to a database."""
        pass
