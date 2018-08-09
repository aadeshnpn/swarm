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

    def tns_connect(self):
        """Connect with tnscon string."""
        try:
            connect = pgsql.connect(
                database=self.dbname, user=self.username, password=self.passwd,
                host=self.hostname, port=self.sport)
        except:
            print (
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
            print ("Code:", e.pgcode)
            print ("Message:", e.pgerror)
            print (sys.exc_info(), "Function name:execute_statement")
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
            print ("Code:", e.pgcode)
            print ("Message:", e.pgerror)
            print (sys.exc_info(), "Function name:execute_statement_bind")
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
        for k, v in kwargs.iteritems():
            print ("%s =%s" % (k, v))
            if k == "function_name":
                functname = v
        try:
            print ("Function name:", functname, "Function Args:", funct_args)
            # logger.info("Procedure arguments:"+proc_args)
            output = self.cursor.callproc(functname, funct_args)
            # output = output.fetchall()
        except:
            print ("Function error", sys.exc_info())
            return False
        else:
            self.con.commit()
            return int(output)

    def execute_proc(self, *args, **kwargs):
        """Execute a pl/sql procedure with variable args."""
        proc_args = []
        for a in args:
            print (a)
            proc_args.append(a)
        for k, v in kwargs.iteritems():
            print ("%s =%s" % (k, v))
            if k == "proc_name":
                procname = v
        try:
            print ("Proc Args:", proc_args)
            # logger.info("Procedure arguments:"+proc_args)
            self.cursor.callproc(procname, proc_args)
        except:
            print "Procedure error"
            return False
        else:
            self.con.commit()
            return True

    def excute_spool(self, **kwargs):
        """Execute pl/sql spool."""
        for k, v in kwargs.iteritems():
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

    def insertInfo(self, session_id, server_ip, location, client, machine):
        """Insert info."""
        exestat = Execute(self.conn)
        # funct_name='public.f_insert_session_info'
        output = 0
        try:
            exestat.cursor.execute("""INSERT INTO session_info(id,serverip,
                request_location,request_client,machine_type) VALUES (
                    %s,%s,%s,%s,%s);""", (
                session_id, server_ip, location, client, machine))
            output = exestat.cursor.execute(
                "SELECT sn from session_info where id=" + "'" + session_id +
                "'")
            output = exestat.cursor.fetchall()
            self.conn.commit()
            exestat.close()
        except:
            print ("Unexpected error function insertInfo:", sys.exc_info())
            return False
        else:
            return int(output[0][0])

    def insertDetails(
            self, session_sn, query, response, timetaken, training_flag,
            verified):
        """Insert details into db."""
        exestat = Execute(self.conn)
        try:
            exestat.cursor.execute(
                """INSERT INTO session_details(
                    session_info_sn, query, response, timetaken, training_flag,
                    verified) VALUES (%s,%s,%s,%s,%s,%s);""", (
                    session_sn, query, response, timetaken, training_flag,
                    verified)
                )
            self.conn.commit()
            exestat.close()
        except:
            print ("Unexpected error function insertDetails:", sys.exc_info())
            return False
        else:
            return True

    def retrieve_info(self, query):
        """Reterive info from the db."""
        cur = self.conn.cursor()
        try:
            cur.execute(query)
            data = cur.fetchall()
            self.conn.commit()
            self.conn.close()
            return data
        except:
            print ("Unexptected error function:", sys.exc_info())
            return False

    def update_table_info(self, query):
        """Update the table in db."""
        exestat = Execute(self.conn)
        try:
            exestat.cursor.execute(query)
            self.conn.commit()
            exestat.close()
        except:
            print ("Unexpected error function update info:", sys.exc_info())
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
        except:
            print ("Unexpected error function execute query:", sys.exc_info())
            return False
        else:
            return True
