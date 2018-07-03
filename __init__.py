# -*- coding: utf-8 -*-
"""
Swarm Agent-Based Modeling Framework

"""
import datetime

from .swarms.lib.space import Grid
from .swarms.lib.time import (
    SimultaneousActivation, RandomActivation, StagedActivation
    )
from .swarms.agent import SwarmAgent
from .swarms.model import EnvironmentModel
from .swarms.utils.jsonhandler import JsonData


__all__ = ["Grid", "SwarmAgent", "EnvironmentModel", "SimultaneousActivation", "RandomActivation", "StagedActivation", "JsonData"]

__title__ = 'swarms'
__version__ = '0.0.1'
__license__ = 'Apache 2.0'
__copyright__ = 'Copyright %s Project swarm Team' % datetime.date.today().year
