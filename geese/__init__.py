# -*- coding: utf-8 -*-
"""
Swarm Agent-Based Modeling Framework

"""
import datetime

from .lib.space import Grid
from .lib.time import SimultaneousActivation, RandomActivation, StagedActivation
from .swarms.agent import SwarmAgent
from .swarms.model import EnvironmentModel


__all__ = ["Grid", "SwarmAgent", "EnvironmentModel", "SimultaneousActivation", "RandomActivation", "StagedActivation"]

__title__ = 'swarm'
__version__ = '0.0.1'
__license__ = 'Apache 2.0'
__copyright__ = 'Copyright %s Project swarm Team' % datetime.date.today().year
