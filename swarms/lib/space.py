# -*- coding: utf-8 -*-
"""
The grid class for swarm framework.

Grid: base grid, a simple dictionary.

"""
import math
import numpy as np
import re


class Grid:
    """Grid class.

    Class that defines grid strucutre having attribute
    Width, height and grid size.
    """

    # pylint: disable=too-many-instance-attributes
    # Nine is reasonable in this grid class

    def __init__(self, width, height, grid_size=10):
        """Constructors for grid.

        Args:
            width: total width
            height: total height
            grid_size: granularity of the size of grid

        Attributes:
            x_limit: x-axis length in both direction
            y_limit: y-axis length in both direction

            grid: dictionary object which value is adjacent points
                    of grid and its value is the grid name

            grid_objects: dictionary object which value is the grid name
                            and its value is the list of environment objects

        """
        self.width = width
        self.height = height
        self.x_limit = width / 2
        self.y_limit = height / 2
        self.grid_size = grid_size
        self.grid = dict()
        self.grid_objects = dict()
        # self.width_fix = int(self.x_limit % self.grid_size)
        # self.height_fix = int(self.y_limit % self.grid_size)

        # If the width or height is not comptiable with grid size
        if self.x_limit % self.grid_size != 0 \
                or self.y_limit % self.grid_size != 0:
                    print("Grid size invalid")
                    exit(1)

        # Create list for x cordinate & y cordinate to create grid
        list_xcords = np.arange(
            -self.width / 2, self.width / 2, self.grid_size).tolist()
        list_ycords = np.arange(
            -self.height / 2, self.height / 2, self.grid_size).tolist()

        indx = 1
        for ycord in list_ycords:
            for xcord in list_xcords:
                x_1 = xcord
                y_1 = ycord
                x_2 = xcord + self.grid_size
                y_2 = ycord + self.grid_size
                self.grid[(x_1, y_1), (x_2, y_2)] = indx
                self.grid_objects[indx] = []
                indx += 1

        self.grid_len = indx - 1

    def modify_points(self, point):
        """Modify poitns if the location line in the grid line."""
        x, y = point[0], point[1]

        if point[0] % self.grid_size == 0:
            x = point[0] + 1
        if point[1] % self.grid_size == 0:
            y = point[1] + 1

        if point[0] >= self.x_limit:
            x = point[0] - self.grid_size + 1

        if point[1] >= self.y_limit:
            y = point[1] - self.grid_size + 1
        return (x, y)

    def find_lowerbound(self, point):
        """Find the lower bound from the point."""
        point = self.find_upperbound(point)
        return (point[0] - self.grid_size, point[1] - self.grid_size)

    def find_upperbound(self, point):
        """Find the upper bound from the point."""
        point = self.modify_points(point)
        return (point[0] + self.grid_size - 1 * (
            point[0] % self.grid_size), point[1] + self.grid_size - 1 * (
                point[1] % self.grid_size))

    def find_grid(self, point):
        """Find the grid based on the point passed."""
        grid_key = (self.find_lowerbound(point), self.find_upperbound(point))
        try:
            return grid_key, self.grid[grid_key]
        except KeyError:
            print('KeyError', 'No grid key for ', grid_key)
            exit()

    def get_horizontal_neighbours(self, center_grid, scale, width_scale):
        """Get the neighboring horizontal grids."""
        valid_horizontal_start = (math.floor(
            (center_grid - 1) / width_scale) * width_scale) + 1
        valid_horizontal_end = math.ceil(
            center_grid / width_scale) * width_scale
        if(center_grid - scale) < valid_horizontal_start:
            horizontal_start = valid_horizontal_start
        else:
            horizontal_start = center_grid - scale
        if(center_grid + scale + 1) > valid_horizontal_end:
            horizontal_end = valid_horizontal_end + 1
        else:
            horizontal_end = center_grid + scale + 1
        horizontal_grid = list(range(horizontal_start, horizontal_end, 1))
        return horizontal_grid

    # Find the adjacent grid based on radius
    def get_neighborhood(self, point, radius):
        """Get the neighboring grids."""
        all_grid = []
        center_grid_key, center_grid = self.find_grid(point)

        if self.grid_size >= radius:
            return [center_grid]
        else:
            scale = int(radius / self.grid_size)
            width_scale = int(self.width / self.grid_size)
            horizontal_grid = self.get_horizontal_neighbours(
                center_grid, scale, width_scale)
            vertical_grid = list(range(
                center_grid - scale * width_scale, center_grid +
                1 + scale * width_scale, width_scale))
            h_v_grid = []
            for grid in vertical_grid:
                h_v_grid += self.get_horizontal_neighbours(
                    grid, scale, width_scale)
            all_grid = h_v_grid + horizontal_grid
            all_grid = [grid for grid in all_grid if grid > 0 and
                        grid <= self.grid_len]
        return list(set(all_grid))


    def add_object_to_grid(self, point, objects):
        """Add object to a given grid."""
        grid_values = self.get_neighborhood(point, objects.radius)
        for grid in grid_values:
            # gridobjects = self.get_objects(None, grid)
            # for gobject in gridobjects:
            #     if not re.match('.*Agent.*' , type(gobject).__name__):
            #         if gobject.deathable and re.match('.*Agent.*' , type(objects).__name__):
            #             objects.dead = True
            # print(grid, objects)
            self.grid_objects[grid].append(objects)

    # Remove object to the given grid
    def remove_object_from_grid(self, point, objects):
        """Remove object from the given grid."""
        grid_values = self.get_neighborhood(point, objects.radius)
        for grid in grid_values:
            self.grid_objects[grid].remove(objects)

    def move_object(self, point, objects, newpoint):
        """Move object from the give grid to new grid."""
        grid_key, grid_value = self.find_grid(point)
        new_grid_key, new_grid_value = self.find_grid(newpoint)
        if grid_value != new_grid_value:
            if re.match('.*Agent.*' , type(objects).__name__) and objects.dead:
                return False
            elif re.match('.*Agent.*' , type(objects).__name__) and not objects.dead:
                # print(point, newpoint, grid_value, new_grid_value)
                if self.check_grid_deathable_constraints(new_grid_value):
                    objects.dead = True
                    objects.moveable = False
                    self.remove_object_from_grid(point, objects)
                    self.add_object_to_grid(newpoint, objects)
                    return True
                else:
                    if self.check_grid_objects_constraints(new_grid_value):
                        self.remove_object_from_grid(point, objects)
                        self.add_object_to_grid(newpoint, objects)
                        return True
                    else:
                        return False
            else:
                if self.check_grid_objects_constraints(new_grid_value):
                    self.remove_object_from_grid(point, objects)
                    self.add_object_to_grid(newpoint, objects)
                    return True
                else:
                    return False
        else:
            return True

    # Check limits for the environment boundary
    def check_limits(self, i, d):
        """Check the location is valid."""
        x, y = i
        if x > (self.width / 2):
            x = x - (x - self.x_limit) - 2
            d = np.pi + d
        elif x < (self.width / 2) * -1:
            x = x - (x + self.x_limit) + 2
            d = np.pi + d
        if y > (self.height / 2):
            y = y - (y - self.y_limit) - 2
            d = np.pi + d
        elif y < (self.height / 2) * -1:
            y = y - (y + self.y_limit) + 2
            d = np.pi + d
        return ((int(x), int(y)), d)

    def check_grid_objects_constraints(self, new_grid_value):
        """Check the constraints on the next location."""
        # grid_key, grid_value = self.find_grid(source_obj.location)
        # new_grid_key, new_grid_value = self.find_grid(next_loc)
        passable = True
        objects_in_next_grid = self.get_objects(None, new_grid_value)
        for obj in objects_in_next_grid:
            if not obj.passable:
                passable = False
                break
        return passable

    def check_grid_deathable_constraints(self, new_grid_value):
        """Check the constraints on the next location."""
        # grid_key, grid_value = self.find_grid(source_obj.location)
        # new_grid_key, new_grid_value = self.find_grid(next_loc)
        dead = False
        objects_in_next_grid = self.get_objects(None, new_grid_value)
        for obj in objects_in_next_grid:
            try:
                if obj.deathable:
                    dead = True
                    break
            except:
                pass
        return dead

    # Using fancy search to find the object in the particular grid
    def get_objects(self, object_name, grid_value):
        """Use fancy search to find objects in a grid."""
        if object_name is not None:
            return list(filter(
                lambda x: type(x).__name__ == object_name,
                self.grid_objects[grid_value]))
        else:
            return list(filter(
                lambda x: type(x).__name__ != 'list',
                self.grid_objects[grid_value]))

    def get_objects_from_grid(self, object_name, point):
        """Get objects from grid given a location."""
        grid_key, grid_value = self.find_grid(point)
        return self.get_objects(object_name, grid_value)

    def get_objects_from_list_of_grid(self, object_name, grid_list):
        """Get list of objects from grid list."""
        object_list = []
        for grid in grid_list:
            object_list += self.get_objects(object_name, grid)
        return object_list
