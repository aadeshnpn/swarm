"""Store the information from the experiments."""

from pathlib import Path
from swarms.utils.db import Dbexecute


class Experiment:
    """Experiment class.

    This class corresponds to the experiment table in db.
    """

    def __init__(self, connect, runid):
        """Constructor."""

        self.connect = connect
        self.runid = runid
        self.sn = None

    def insert_experiment(self):
        """Call db function to insert record into db."""
        dbexec = Dbexecute(self.connect)
        self.sn = dbexec.insert_experiment(self.runid)

    def update_experiment(self):
        """Update enddate column."""
        dbexec = Dbexecute(self.connect)

        # Update end time
        dbexec.execute_query(
            "UPDATE experiment \
            set end_date=timezone('utc'::text, now()) where sn=" + str(self.sn))


class Results:
    """Define the results atrributes.

    This class defines the common attributes for each experiments for each
    agents.
    """

    def __init__(
        self, foldername, connect, id, agent_name, step, timestep, beta, fitness, diversity,
        explore, foraging, neighbour, genotype, phenotype, bt
            ):
        """Initialize the attributes."""
        self.foldername = foldername
        self.connect = connect
        self.context = {
            "id": id,
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
            "bt": "bt"
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
        # First check if the id is present
        data = list(self.context.values())
        print('exp details data', data)
        dbexec = Dbexecute(self.connect)
        dbexec.insert_experiment_best(data)


class Best:
    """Define the results atrributes.

    This class defines the best attributes for each experiments for all
    agents.
    """

    def __init__(
        self, foldername, connect, id, agent_name, header, step, beta, fitness, diversity,
        explore, foraging, phenotype
            ):
        """Initialize the attributes."""
        self.foldername = foldername
        self.connect = connect

        self.context = {
            "id": id,
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

    def save(self):
        """Save to both medium."""
        self.save_to_file()
        self.save_to_db()

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
        data = list(self.context.values())
        print('best data', data, self.context.keys())
        dbexec = Dbexecute(self.connect)
        dbexec.insert_experiment_best(data)
