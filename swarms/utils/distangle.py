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


"""
From Algorithm tutor
https://algorithmtutor.com/Computational-Geometry/Determining-if-two-consecutive-segments-turn-left-or-right/
"""


class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def subtract(self, p):
    	return Point(self.x - p.x, self.y - p.y)

    def __str__(self):
        return '(' + str(self.x) + ', ' + str(self.y) + ')'


def cross_product(p1, p2):
	return p1.x * p2.y - p2.x * p1.y


def direction(p1, p2, p3):
	return  cross_product(p3.subtract(p1), p2.subtract(p1))


def collinear(p1, p2, p3):
	return direction(p1, p2, p3) == 0


def right(p1, p2, p3):
	return direction(p1, p2, p3) > 0


def left(p1, p2, p3):
	return direction(p1, p2, p3) < 0


# checks if p lies on the segment p1p2
def on_segment(p1, p2, p):
    return min(p1.x, p2.x) <= p.x <= max(p1.x, p2.x) and min(p1.y, p2.y) <= p.y <= max(p1.y, p2.y)


# checks if line segment p1p2 and p3p4 intersect
def check_intersect(p1, p2, p3, p4):
    p1 = Point(p1[0], p1[1])
    p2 = Point(p2[0], p2[1])
    p3 = Point(p3[0], p3[1])
    p4 = Point(p4[0], p4[1])
    d1 = direction(p3, p4, p1)
    d2 = direction(p3, p4, p2)
    d3 = direction(p1, p2, p3)
    d4 = direction(p1, p2, p4)

    if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and \
        ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
        return True

    elif d1 == 0 and on_segment(p3, p4, p1):
        return True
    elif d2 == 0 and on_segment(p3, p4, p2):
        return True
    elif d3 == 0 and on_segment(p1, p2, p3):
        return True
    elif d4 == 0 and on_segment(p1, p2, p4):
        return True
    else:
        return False