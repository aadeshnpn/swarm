# -*- coding: utf-8 -*-
"""
Grid: base grid, a simple dictionary.

"""

import itertools
import numpy as np
import random
import math

class Grid:
    
    def __init__(self, width, height, grid_size=10):
        self.width = width
        self.height = height
        self.x_limit = width / 2
        self.y_limit = height / 2
        self.grid_size = grid_size
        self.grid = {}
        self.grid_objects = {}
        self.width_fix = int (self.x_limit % self.grid_size)
        self.height_fix = int (self.y_limit % self.grid_size)
        # If the width or height is not comptiable with grid size
        if self.x_limit % self.grid_size != 0 or self.y_limit % self.grid_size != 0:
            print ("Grid size invalid")
            exit(1)

        # Create list for x cordinate & y cordinate to create grid
        list_xcords = np.arange(-self.width/2, self.width/2, self.grid_size).tolist() 
        list_ycords = np.arange(-self.height/2, self.height/2, self.grid_size).tolist()
        
        # Create grid structure 
        i = 1
        for ycord in list_ycords:
            for xcord in list_xcords:
                x1 = xcord; y1 = ycord; x2 = xcord + self.grid_size; y2 = ycord + self.grid_size
                self.grid[(x1,y1),(x2,y2)] = i
                self.grid_objects[i] = []
                i += 1
        self.grid_len = i - 1        


    ## Modify poitns if the location line in the grid line
    def modify_points(self, point):
        x, y = point[0], point[1]

        if point[0] % self.grid_size == 0:
            x = point[0] + 1
        if point[1] % self.grid_size == 0:
            y = point[1] + 1

        if point[0] >= self.x_limit:
            x = point[0] - self.grid_size

        if point[1] >= self.y_limit:
            y = point[1] - self.grid_size
        return (x, y)

    ## Find the lower bound from the point
    def find_lowerbound(self, point):
        point = self.find_upperbound(point)
        return (point[0] - self.grid_size, point[1] - self.grid_size)

    ## Find the upper bound from the point
    def find_upperbound(self, point):
        point = self.modify_points(point)    
        return (point[0] + self.grid_size - 1 * (point[0] % self.grid_size), point[1] + self.grid_size - 1 * (point[1] % self.grid_size))

    ## Find the grid based on the point passed
    def find_grid(self, point):
        grid_key = (self.find_lowerbound(point), self.find_upperbound(point))
        try:
            return grid_key, self.grid[grid_key]
        except KeyError:
            return None, None

    ## Find the adjacent grid based on radius
    def get_neighborhood(self, point, radius):
        all_grid = []
        center_grid_key,center_grid = self.find_grid(point)
        if self.grid_size > radius:
            return [center_grid]
        else:
            scale = math.ceil(radius/self.grid_size)
            horizontal_grid = list(range(center_grid-scale,center_grid+scale,1))
            width_scale = int(self.width / self.grid_size)
            vertical_grid = list(range(center_grid-scale*width_scale,center_grid+scale*width_scale,width_scale))
            h_v_grid = []
            for grid in vertical_grid:
                h_v_grid += list(range(grid-scale,grid+scale,1))
            all_grid = h_v_grid + horizontal_grid
            all_grid = [grid for grid in all_grid if grid > 0 and grid <= self.grid_len]
        return list(set(all_grid))

    ## Add agent to the given grid
    def add_object_to_grid(self, point, agent):
        grid_key, grid_value = self.find_grid(point)        
        self.grid_objects[grid_value].append(agent)
        agent.location = point

    ## Remove agent to the given grid
    def remove_object_from_grid(self, point, agent):
        grid_key, grid_value = self.find_grid(point)        
        self.grid_objects[grid_val].remove(agent)


    def move_object(self, point, object, newpoint):
        pass
        
    ## Check limits for the environment boundary
    def check_limits(self,i,d):  
        x, y = i 
        if x > (self.width/2):
            x = x - (x - self.x_limit) - 2
            d = np.pi + d
        elif x < (self.width/2)  * -1:
            x = x - (x + self.x_limit) + 2
            d = np.pi + d
        if y > (self.height/2):
            y = y - (y - self.y_limit) - 2
            d = np.pi + d            
        elif y < (self.height/2) * -1:
            y = y - (y + self.y_limit) + 2
            d = np.pi + d            
        return ((x,y),d)

    ## Using fancy search to find the obstacles object in the particular grid
    def get_objects(self, object_name, point):
        grid_key, grid_value = self.find_grid(point)        
        return list(filter(lambda x : type(x).__name__ == object_name, self.grid_objects[grid_value]))          
