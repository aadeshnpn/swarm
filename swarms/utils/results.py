"""Store the information from the experiments."""

from pathlib import Path
from swarms.utils.db import Dbexecute
from collections import OrderedDict
import time


class Experiment:
    """Experiment class.

    This class corresponds to the experiment table in db.
    """

    def __init__(
            self, connect, runid, N, seed, expname, iter, width, height, grid,
            phenotype=None, db=False):
        """Constructor."""
        self.connect = connect
        self.runid = runid
        self.sn = 1
        self.N = N
        self.seed = seed
        self.expname = expname
        self.iter = iter
        self.width = width
        self.height = height
        self.grid = grid
        self.phenotype = phenotype
        self.db_flag = db

    def insert_experiment(self):
        """Call db function to insert record into db."""
        if self.db_flag:
            try:
                dbexec = Dbexecute(self.connect)
                self.sn = dbexec.insert_experiment(
                    self.runid, self.N, self.seed, self.expname, self.iter, self.width,
                    self.height, self.grid)
            except Exception as err:
                pass
        else:
            self.sn = 1

    def insert_experiment_simulation(self):
        """Call db function to insert record into db."""
        if self.db_flag:
            try:
                dbexec = Dbexecute(self.connect)
                self.sn = dbexec.insert_experiment_simulation(
                    self.runid, self.N, self.seed, self.expname, self.iter, self.width,
                    self.height, self.grid, self.phenotype)
            except Exception as err:
                pass
        else:
            self.sn = 1

    def update_experiment(self):
        """Update enddate column."""
        if self.db_flag:
            try:
                dbexec = Dbexecute(self.connect)
                # Update end time
                dbexec.execute_query(
                    "UPDATE experiment \
                    set end_date=timezone('utc'::text, now()) where sn=" + str(
                        self.sn))
            except Exception as err:
                pass
        else:
            self.sn = 1

    def update_experiment_simulation(self, value, sucess):
        """Update enddate column."""
        if self.db_flag:
            try:
                dbexec = Dbexecute(self.connect)
                # Update end time
                dbexec.execute_query(
                    "UPDATE experiment \
                    set end_date=timezone('utc'::text, now()), total_value=\
                    " + str(value) + ", sucess=" + str(sucess) + "\
                    where sn=" + str(self.sn))
            except Exception as err:
                pass
        else:
            self.sn = 1


class Results:
    """Define the results atrributes.

    This class defines the common attributes for each experiments for each
    agents.
    """

    def __init__(
        self, foldername, connect, id, agent_name, step, timestep, beta,
        fitness, diversity, explore, foraging, postcond,
        constraints, selector, genotype, phenotype, bt):
        """Initialize the attributes."""
        self.foldername = foldername
        self.connect = connect
        self.context = OrderedDict([
            ("id", id),
            ("name", int(agent_name)),
            ("step", step),
            ("timestep", timestep),
            ("beta", float(beta)),
            ("fitness", float(fitness)),
            ("diversity", float(diversity)),
            ("explore", float(explore)),
            ("foragingp", float(foraging)),
            ("postcond", float(postcond)),
            ("constraints", float(constraints)),
            ("selector", float(selector)),
            ("genotype", genotype),
            ("phenotype", phenotype),
            ("bt", str(bt))
        ])
        # self.template = """
        # Id, Agent Name, Step, Beta, Fitness, Diversity, Explore, Foraging,\
        # Neighbours, Genotype, Phenotype, BT
        self.template = """{id}|{name}|{step}|{timestep}|{beta}|{fitness}|{diversity}|{explore}|{foragingp}|{postcond}|{constraints}|{selector}|{genotype}|{phenotype}|{bt}
        """
        # Write a header to the file for pandas dataframe
        self.header = """id|name|step|timestep|beta|fitness|diversity|explore|foragingp|postcond|constraints|selctor|genotype|phenotype|bt\n
        """

    def save_to_file(self):
        """Save results to a flat file."""
        filename = self.foldername + '/' + 'results.csv'
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
        dbexec = Dbexecute(self.connect)
        dbexec.insert_experiment_details(data)


class Best:
    """Define the results atrributes.

    This class defines the best attributes for each experiments for all
    agents.
    """

    def __init__(
        self, foldername, connect, id, agent_name, header, step, beta, fitness,
        diversity, explore, foraging, postcond, constraints, selector, genotype, phenotype, db=False
            ):
        """Initialize the attributes."""
        self.foldername = foldername
        self.connect = connect
        self.db_falg = db
        self.context = OrderedDict([
            ("id", id),
            ("name", int(agent_name)),
            ("header", header),
            ("step", step),
            ("beta", round(float(beta), 4)),
            ("fitness", round(float(fitness), 4)),
            ("diversity", round(float(diversity), 4)),
            ("explore", round(float(explore), 4)),
            ("foragingp", round(float(foraging), 4)),
            ("postcond", round(float(postcond), 4)),
            ("constraints", round(float(constraints), 4)),
            ("selector", round(float(selector), 4)),
            ("genotype", genotype),
            ("phenotype", phenotype)
        ])

        self.template = """{id}|{header}|{name}|{step}|{beta}|{fitness}|{diversity}|{explore}|{foragingp}|{postcond}|{constraints}|{selector}|{genotype}|{phenotype}
        """
        # Write a header to the file for pandas dataframe
        self.header = """id|header|name|step|beta|fitness|diversity|explore|foragingp|postcond|constraints|selctor|genotype|phenotype\n
        """

    def save(self):
        """Save to both medium."""
        self.save_to_file()
        if self.db_falg:
            try:
                self.save_to_db()
            except Exception as err:
                pass

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
        dbexec = Dbexecute(self.connect)
        dbexec.insert_experiment_best(data)


class SimulationResults:
    """Define the simluation results atrributes.

    This class defines the best attributes for each experiments for all
    agents.
    """

    def __init__(
        self, foldername, connect, id, step, fitness, phenotype, db=False
            ):
        """Initialize the attributes."""
        self.foldername = foldername
        self.connect = connect
        self.phenotype = phenotype
        self.db_falg = db
        self.context = OrderedDict([
            ("id", id),
            ("step", step),
            ("fitness", float(fitness))
        ])

        self.template = """{id}|{step}|{fitness}
        """
        # Write a header to the file for pandas dataframe
        self.header = """id|step|fitness\n
        """

    def save(self):
        """Save to both medium."""
        self.save_to_file()
        if self.db_falg:
            try:
                self.save_to_db()
            except Exception as err:
                pass

    def save_phenotype(self):
        """Save the phenotype to a separate file."""
        fname = self.foldername + '/' + 'phenotype.txt'
        with open(fname, 'w') as pfile:
            pfile.write(self.phenotype)

    def save_to_file(self):
        """Save results to a flat file."""
        filename = self.foldername + '/' + 'simulation.csv'
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
        dbexec = Dbexecute(self.connect)
        dbexec.insert_experiment_best(data)


class SimulationResultsTraps:
    """Define the simluation results atrributes.

    This class defines the best attributes for each experiments for all
    agents.
    """

    def __init__(
        self, foldername, connect, id, step, fitness, phenotype, deadagent, db=False
            ):
        """Initialize the attributes."""
        self.foldername = foldername
        self.connect = connect
        self.phenotype = phenotype
        self.deadagent = deadagent
        self.db_flag = db

        self.context = OrderedDict([
            ("id", id),
            ("step", step),
            ("fitness", float(fitness)),
            ("deadagent", deadagent),
        ])

        self.template = """{id}|{step}|{fitness}|{deadagent}
        """
        # Write a header to the file for pandas dataframe
        self.header = """id|step|fitness|deadagent\n
        """

    def save(self):
        """Save to both medium."""
        self.save_to_file()
        if self.db_flag:
            try:
                self.save_to_db()
            except Exception as Err:
                pass

    def save_phenotype(self):
        """Save the phenotype to a separate file."""
        fname = self.foldername + '/' + 'phenotype.txt'
        with open(fname, 'w') as pfile:
            pfile.write(self.phenotype)

    def save_to_file(self):
        """Save results to a flat file."""
        filename = self.foldername + '/' + 'simulation.csv'
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
        dbexec = Dbexecute(self.connect)
        dbexec.insert_experiment_best(data)


class SimulationResultsLt:
    """Define the simluation results atrributes.

    This class defines the best attributes for each experiments for all
    agents.
    """

    def __init__(
        self, foldername, connect, id, step, fitness,
        phenotype, deadagent, ltrateavg, ltratestd, grate, db=False
            ):
        """Initialize the attributes."""
        self.foldername = foldername
        self.connect = connect
        self.phenotype = phenotype
        self.deadagent = deadagent
        self.db_flag = db
        self.ltrateavg = ltrateavg
        self.ltratestd = ltratestd
        self.grate = grate

        self.context = OrderedDict([
            ("id", id),
            ("step", step),
            ("fitness", float(fitness)),
            ("deadagent", deadagent),
            ("ltrateavg", ltrateavg),
            ("ltratestd", ltratestd),
            ("grate", grate),
        ])

        self.template = """{id}|{step}|{fitness}|{deadagent}|{ltrateavg}|{ltratestd}|{grate}
        """
        # Write a header to the file for pandas dataframe
        self.header = """id|step|fitness|deadagent|ltrateavg|ltratestd|grate\n
        """

    def save(self):
        """Save to both medium."""
        self.save_to_file()
        if self.db_flag:
            try:
                self.save_to_db()
            except Exception as Err:
                pass

    def save_phenotype(self):
        """Save the phenotype to a separate file."""
        fname = self.foldername + '/' + 'phenotype.txt'
        with open(fname, 'w') as pfile:
            pfile.write(self.phenotype)

    def save_to_file(self):
        """Save results to a flat file."""
        filename = self.foldername + '/' + 'simulation.csv'
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
        dbexec = Dbexecute(self.connect)
        dbexec.insert_experiment_best(data)


class SimulationResultsWTime:
    """Define the simluation results atrributes.

    This class defines the best attributes for each experiments for all
    agents.
    """

    def __init__(
        self, foldername, connect, id, step, fitness, phenotype
            ):
        """Initialize the attributes."""
        self.foldername = foldername
        self.connect = connect
        self.phenotype = phenotype
        curr_time = time.time()
        self.context = OrderedDict([
            ("id", id),
            ("step", step),
            ("fitness", float(fitness)),
            ("time", curr_time)
        ])

        self.template = """{id}|{step}|{fitness}|{time}
        """
        # Write a header to the file for pandas dataframe
        self.header = """id|step|fitness|time\n
        """

    def save(self):
        """Save to both medium."""
        self.save_to_file()
        # self.save_to_db()

    def save_phenotype(self):
        """Save the phenotype to a separate file."""
        fname = self.foldername + '/' + 'phenotype.txt'
        with open(fname, 'w') as pfile:
            pfile.write(self.phenotype)

    def save_to_file(self):
        """Save results to a flat file."""
        filename = self.foldername + '/' + 'simulation.csv'
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
        dbexec = Dbexecute(self.connect)
        dbexec.insert_experiment_best(data)
