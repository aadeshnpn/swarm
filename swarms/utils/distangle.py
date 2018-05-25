"""Collection of math modules."""

import numpy as np
import math


def distance(x1, y1, x2, y2):
    """Compute distnace between two points."""
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def point_distance(u, v):
    """Compute distance."""
    return distance(u[0], u[1], v[0], v[1])


def safe_angle(a, b):
    """Compute safe angle."""
    angle = np.arctan2(b[1], b[0]) - np.arctan2(a[1], a[0])
    return angle


def get_direction(x, y):
    """Compute direction between two points."""
    dx = x[0] - y[0]
    dy = x[1] - y[1]
    return np.arctan2(dy, dx)
