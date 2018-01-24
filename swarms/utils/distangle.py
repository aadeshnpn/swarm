import numpy as np
import math


def distance(x1, y1, x2, y2):
	return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def point_distance(u, v):
	return distance(u[0], u[1], v[0], v[1])


def safe_angle(a, b):
    angle = np.arctan2(b[1], b[0]) - np.arctan2(a[1], a[0])
    return angle


def get_direction(x, y):
    dx = x[0] - y[0]
    dy = x[1] - y[1]
    return np.arctan2(dy, dx)
