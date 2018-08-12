"""Script for basic db connection to postgresql.

#@Author:Aadeshnpn

"""

import sys
import psycopg2 as pgsql


class Connect():
    """Class to connect DB."""

    def __init__(
            self, dbname, username, passwd, hostname, sport=5432,
            racflag=False):
        """Constructor."""
        self.dbname = dbname
        self.username = username
        self.passwd = passwd
        self.hostname = hostname
        self.sport = sport
        self.racflag = racflag
        self.__name__ = 'Connect'

    def tns_connect(self):
        """Connect with tnscon string."""
        try:
            connect = pgsql.connect(
                database=self.dbname, user=self.username, password=self.passwd,
                host=self.hostname, port=self.sport)
        except pgsql.DatabaseError:
            print(
                "Unexpected error:", sys.exc_info(), "Function name:",
                self.__name__)
        else:
            return connect


class Execute():
    """Class to execute different oracle statements."""

    def __init__(self, con):
        """Constructor."""
        self.con = con
        self.cursor = self.con.cursor()

    def execute_statement(self, statement):
        """Execute pgsql statement."""
        try:
            self.cursor.execute(statement)
        except pgsql.Error as e:
            # error,=e.args
            print("Code:", e.pgcode)
            print("Message:", e.pgerror)
            print(sys.exc_info(), "Function name:execute_statement")
            self.con.rollback()
        else:
            self.con.commit()
            return self.cursor

    def execute_statement_bind(self, statement, bindvars):
        """Execute oracle statement using bind vars."""
        try:
            self.cursor.execute(statement, bindvars)
        except pgsql.Error as e:
            # error,= e.args
            print("Code:", e.pgcode)
            print("Message:", e.pgerror)
            print(sys.exc_info(), "Function name:execute_statement_bind")
            self.con.rollback()
        else:
            self.con.commit()
            return bindvars['sn']

    def execute_function(self, *args, **kwargs):
        """Execute a pl/sql function with variable args."""
        funct_args = []
        for a in args:
            # print a
            funct_args.append(a)
        for k, v in kwargs.items():
            print("%s =%s" % (k, v))
            if k == "function_name":
                functname = v
        try:
            print("Function name:", functname, "Function Args:", funct_args)
            # logger.info("Procedure arguments:"+proc_args)
            output = self.cursor.callproc(functname, funct_args)
            # output = output.fetchall()
        except pgsql.DatabaseError:
            print("Function error", sys.exc_info())
            return False
        else:
            self.con.commit()
            return int(output)

    def execute_proc(self, *args, **kwargs):
        """Execute a pl/sql procedure with variable args."""
        proc_args = []
        for a in args:
            print(a)
            proc_args.append(a)
        for k, v in kwargs.items():
            print("%s =%s" % (k, v))
            if k == "proc_name":
                procname = v
        try:
            print("Proc Args:", proc_args)
            # logger.info("Procedure arguments:"+proc_args)
            self.cursor.callproc(procname, proc_args)
        except pgsql.DatabaseError:
            print("Procedure error")
            return False
        else:
            self.con.commit()
            return True

    def excute_spool(self, **kwargs):
        """Execute pl/sql spool."""
        for k, v in kwargs.items():
            if k == "bulkid":
                # bulkid = v
                pass
            elif k == "spoolname":
                # spoolname = v
                pass
            elif k == "query":
                query = v
        # stat = "SELECT "
        output = self.execute_statement(query)
        return output

    def close(self):
        """Close the db cursor."""
        self.cursor.close()


class Trimstrdb():
    """Preprocess the queries."""

    @staticmethod
    def trimdb(inputstr, trimlen=3999):
        """Trim query method.

        Trips the query to 3999 character and replacs single quotes.
        """
        trimstr = inputstr[0:trimlen]
        trimstr = trimstr.replace("'", "''")
        return trimstr


class Dbexecute():
    """Execute queries in DB."""

    def __init__(self, conn):
        """Constructor.

        It takes db connection as argument.
        """
        self.conn = conn
        self.trimstrobj = Trimstrdb()

    def insert_experiment(
            self, id, N, seed, expname, iter, width, height, grid):
        """Insert into experiment table."""
        exestat = Execute(self.conn)
        output = 0
        try:
            exestat.cursor.execute("""INSERT INTO experiment(id, agent_size,
            randomseed, experiment_name, iteration, width, height,
            grid_size) VALUES (
                    %s, %s, %s, %s, %s, %s, %s,
                    %s);""", (id, N, seed, expname, iter, width, height, grid))
            output = exestat.cursor.execute(
                "SELECT sn from experiment where id=" + "'" + str(id) +
                "'")
            output = exestat.cursor.fetchall()
            self.conn.commit()
            exestat.close()
        except pgsql.Error:
            print(
                "Unexpected error function insert_experiment:", sys.exc_info())
            return False
        else:
            return int(output[0][0])

    def insert_experiment_details(self, data_list):
        """Insert into experiment_details table."""
        exestat = Execute(self.conn)
        data = data_list
        try:
            exestat.cursor.execute("""INSERT INTO experiment_details(exp_id,
            agent_name, step, time_step, beta, fitness, diversity,
            explore, forage, neighbours, genotype, phenotype, bt)
            VALUES (
                    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s);""", (
                data[0], data[1], data[2], data[3], data[4], data[5],
                data[6], data[7], data[8], data[9], data[10], data[11],
                data[12])
                )
            # output = exestat.cursor.execute(
            #    "SELECT sn from session_info where id=" + "'" + session_id +
            #    "'")
            # output = exestat.cursor.fetchall()
            self.conn.commit()
            exestat.close()
            return True
        except pgsql.Error:
            print(
                "Unexpected error function insert_experiment_details:",
                sys.exc_info())
            return False
        # else:
        #    return int(output[0][0])

    def insert_experiment_best(self, data_list):
        """Insert into experiment_best table."""
        exestat = Execute(self.conn)
        data = data_list
        try:
            exestat.cursor.execute("""INSERT INTO experiment_best(exp_id,
            agent_name, heading, step, beta, fitness, diversity, explore,
            forage, phenotype) VALUES (
                    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s);""", (
                data[0], data[1], data[2], data[3], data[4], data[5],
                data[6], data[7], data[8], data[9])
                )

            self.conn.commit()
            exestat.close()
            return True
        except pgsql.Error:
            print(
                "Unexpected error function insert_experiment_best:",
                sys.exc_info())
            return False
        # else:
        #    return int(output[0][0])

    def retrieve_info(self, query):
        """Reterive info from the db."""
        cur = self.conn.cursor()
        try:
            cur.execute(query)
            data = cur.fetchall()
            self.conn.commit()
            self.conn.close()
            return data
        except pgsql.Error:
            print("Unexptected error function:", sys.exc_info())
            return False

    def update_table_info(self, query):
        """Update the table in db."""
        exestat = Execute(self.conn)
        try:
            exestat.cursor.execute(query)
            self.conn.commit()
            exestat.close()
        except pgsql.Error:
            print("Unexpected error function update info:", sys.exc_info())
            return False
        else:
            return True

    def execute_query(self, query):
        """Execute a custom query."""
        exestat = Execute(self.conn)
        try:
            exestat.cursor.execute(query)
            self.conn.commit()
            exestat.close()
        except pgsql.Error:
            print("Unexpected error function execute query:", sys.exc_info())
            return False
        else:
            return True
