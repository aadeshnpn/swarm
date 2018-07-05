"""Class defining UI properties."""

from swarms.utils.jsonhandler import JsonData
import time


class UI:
    """UI class."""

    def __init__(
        self, size, hub, agents, sites=None, obstacles=None, traps=None,
            derbis=None, food=None, cues=None):
            """Initialize objects for UI."""
            self.width = size[0]
            self.height = size[1]
            self.hub = hub
            self.agents = agents
            self.sites = sites
            self.obstacles = obstacles
            self.traps = traps
            self.derbis = derbis
            self.food = food
            self.cues = cues

    def step(self):
        """UI function.

        Sending information to UI in a JSON format.
        """
        print(JsonData.to_json(
            self.width, self.height, self.hub, self.sites, self.obstacles,
            self.traps, self.cues, self.food, self.derbis, self.agents)
            )
        time.sleep(0.00833333)
