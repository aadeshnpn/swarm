"""Test class for database."""

import unittest
from swarms.utils.db import Connect, Dbexecute


class TestGrid(unittest.TestCase):
    """Test calss for database."""

    def setUp(self):
        """Set up required stuffs."""
        self.dbname = 'swarm'
        self.username = 'swarm'
        self.passwd = 'swarm'
        self.hostname = 'localhost'
        self.connect = Connect(
            self.dbname, self.username, self.passwd, self.hostname
            )

    def test_connection(self):
        """Test connection to db."""
        # This will return connection object
        tnsconnect = self.connect.tns_connect()

        # Check if the connection is valid
        self.assertEqual(1, tnsconnect.status)

        # Check if its connected to the right db with right parameters
        tns_parms = {
            'host': 'localhost',
            'krbsrvname': 'postgres',
            'options': '',
            'tty': '',
            'dbname': 'swarm',
            'target_session_attrs': 'any',
            'sslmode': 'prefer',
            'port': '5432',
            'user': 'swarm',
            'sslcompression': '1'}

        self.assertDictEqual(tns_parms, tnsconnect.get_dsn_parameters())

    def test_insert_experiment(self):
        """Test insert statement to db."""
        tnsconnect = self.connect.tns_connect()
        dbexec = Dbexecute(tnsconnect)
        sn = dbexec.insert_experiment(20150101024)
        self.assertEqual('int', type(sn).__name__)

        # After insertion is done need to delete the values as well
        retval = dbexec.execute_query("DELETE from experiment where sn=" + str(sn))
        self.assertEqual(True, retval)

    def test_insert_experiment_details(self):
        """Test insert statement for experiment details table.

        Since this table has a foreign key from experiment table. First
        we need to populate experiment table.
        """
        tnsconnect = self.connect.tns_connect()
        dbexec = Dbexecute(tnsconnect)
        sn = dbexec.insert_experiment(20150101025)
        data_list = [sn, 1, 45, 7, 0.99, 80, 78, 45, 2, 5, '[1, 0, 1 ,1, 0]', '<xml>', 'none']
        retval = dbexec.insert_experiment_details(data_list)

        self.assertEqual(True, retval)

        # After insertion is done first need to delete in child table
        retval = dbexec.execute_query(
            "DELETE from experiment_details where exp_id=" + str(sn))

        self.assertEqual(True, retval)
        # After child records deleted, safely delete parent

        retval = dbexec.execute_query(
            "DELETE from experiment where sn=" + str(sn))

        self.assertEqual(True, retval)

    def test_insert_experiment_best(self):
        """Test insert statement for experiment best table.

        Since this table has a foreign key from experiment table. First
        we need to populate experiment table.
        """
        tnsconnect = self.connect.tns_connect()
        dbexec = Dbexecute(tnsconnect)
        sn = dbexec.insert_experiment(20150101025)
        data_list = [sn, 1, 'MEAN', 7, 0.99, 80, 78, 45, 2, '<xml>']
        retval = dbexec.insert_experiment_best(data_list)

        self.assertEqual(True, retval)

        # After insertion is done first need to delete in child table
        retval = dbexec.execute_query(
            "DELETE from experiment_best where exp_id=" + str(sn))

        self.assertEqual(True, retval)
        # After child records deleted, safely delete parent

        retval = dbexec.execute_query(
            "DELETE from experiment where sn=" + str(sn))

        self.assertEqual(True, retval)

    def test_update_experiment(self):
        """Test update statement for experiment table.

        Since this table to update the end time we first need to
        populate the values.
        """
        tnsconnect = self.connect.tns_connect()
        dbexec = Dbexecute(tnsconnect)
        sn = dbexec.insert_experiment(20150101025)

        # Update end time
        retval = dbexec.execute_query(
            "UPDATE experiment \
            set end_date=timezone('utc'::text, now()) where sn=" + str(sn))

        self.assertEqual(True, retval)

        retval = dbexec.execute_query(
            "DELETE from experiment where sn=" + str(sn))

        self.assertEqual(True, retval)
