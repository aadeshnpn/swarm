"""Collection of math modules."""

import numpy as np
import math


def distance(x1, y1, x2, y2):
    """Compute distnace between two points."""
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def point_distance(u, v):
    """Compute distance."""
    return distance(u[0], u[1], v[0], v[1])


def get_direction(x, y):
    """Compute direction between two points."""
    dx = x[0] - y[0]
    dy = x[1] - y[1]
    return np.arctan2(dy, dx)


def round_angles(direction):
    """Round angles such that direction is less that 2*np.pi"""
    return direction % (2 * np.pi)