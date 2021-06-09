# -*- coding: utf-8 -*-
"""
Swarm Agent-Based Modeling Framework

"""
import datetime

from .swarms.lib.space import Grid
from .swarms.lib.time import (
    SimultaneousActivation, RandomActivation, StagedActivation
    )
from .swarms.utils.jsonhandler import JsonData
from .swarms.lib.agent import Agent
from .swarms.lib.model import Model
from .swarms.lib.objects import Hub, Sites

__all__ = [
    "Grid", "Agent", "Model", "SimultaneousActivation",
    "RandomActivation", "StagedActivation", "JsonData",
    "Hub", "Sites"]

__title__ = 'swarms'
__version__ = '1.0.1'
__license__ = 'Apache 2.0'
__copyright__ = 'Copyright %s Project swarm Team' % datetime.date.today().year
