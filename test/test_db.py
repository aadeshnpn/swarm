"""Test class for database."""

import unittest
from swarms.utils.db import Connect


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

    def test_insert(self):
        """Test insert statement to db."""
        pass