'''
Test the grid object
'''
import unittest

from lib.space import Grid

"""
Simple grid for testing
Widht:20, Height:20, Grid:5
 ---- Visual Aid ----
(-10.0, -10.0) (-5.0, -5.0) 1
(-5.0, -10.0) (0.0, -5.0) 2
(0.0, -10.0) (5.0, -5.0) 3
(5.0, -10.0) (10.0, -5.0) 4
(-10.0, -5.0) (-5.0, 0.0) 5
(-5.0, -5.0) (0.0, 0.0) 6
(0.0, -5.0) (5.0, 0.0) 7
(5.0, -5.0) (10.0, 0.0) 8
(-10.0, 0.0) (-5.0, 5.0) 9
(-5.0, 0.0) (0.0, 5.0) 10
(0.0, 0.0) (5.0, 5.0) 11
(5.0, 0.0) (10.0, 5.0) 12
(-10.0, 5.0) (-5.0, 10.0) 13
(-5.0, 5.0) (0.0, 10.0) 14
(0.0, 5.0) (5.0, 10.0) 15
(5.0, 5.0) (10.0, 10.0) 16
"""


class TestGrid(unittest.TestCase):
    '''
    Testing the grid system of swarm framework
    '''

    def setUp(self):
        '''
        Create three grids system with different grid size
        '''
        self.grid_four_equal_width_height = Grid(24, 24, grid_size=4)
        self.grid_four_unequal_width_height = Grid(24, 16, grid_size=4)
        self.grid_five = Grid(20, 20, grid_size=5)

    def test_location_grid_four_equal_width_height(self):
        '''
        For grid 24x24, point (-1,1) should lie in grid number 21
        '''
        grid_key, grid_value = self.grid_four_equal_width_height.find_grid((-1, 1))
        self.assertEqual(grid_value, 21)
        self.assertEqual(grid_key, ((-4, 0), (0, 4)))

        grid_key, grid_value = self.grid_four_equal_width_height.find_grid((4, 4))
        self.assertEqual(grid_value, 29)
        self.assertEqual(grid_key, ((4, 4), (8, 8)))

    def test_location_grid_four_unequal_width_height(self):
        '''
        For grid 24x16, point (-5,1) should lie in grid number 8
        '''
        grid_key, grid_value = self.grid_four_unequal_width_height.find_grid((-5, 1))
        self.assertEqual(grid_value, 14)
        self.assertEqual(grid_key, ((-8, 0), (-4, 4)))

        grid_key, grid_value = self.grid_four_unequal_width_height.find_grid((7, 7))
        self.assertEqual(grid_value, 23)
        self.assertEqual(grid_key, ((4, 4), (8, 8)))

    def test_location_grid_five(self):
        '''
        For grid 20x20, point (-1,4) should lie in grid number 10
        '''
        grid_key, grid_value = self.grid_five.find_grid((-1, 4))
        self.assertEqual(grid_value, 10)
        self.assertEqual(grid_key, ((-5, 0), (0, 5)))

        grid_key, grid_value = self.grid_five.find_grid((9, 9))
        self.assertEqual(grid_value, 16)
        self.assertEqual(grid_key, ((5, 5), (10, 10)))

        grid_key, grid_value = self.grid_five.find_grid((-4, -4))
        self.assertEqual(grid_value, 6)
        self.assertEqual(grid_key, ((-5, -5), (0, 0)))

        grid_key, grid_value = self.grid_five.find_grid((-9, 3))
        self.assertEqual(grid_value, 9)
        self.assertEqual(grid_key, ((-10, 0), (-5, 5)))

    def test_neighbour_grid_equal_width_height(self):
        '''
        Ensure gird gives accurate results for neighbouring
        grids with fixed raidus and fixed point
        '''
        point = (-2, -2)
        radius = 5
        self.assertEqual(self.grid_four_equal_width_height.get_neighborhood(point, radius), [8, 9, 10, 14, 15, 16, 20, 21, 22])

        radius = 9
        self.assertEqual(self.grid_four_equal_width_height.get_neighborhood(point, radius),
        [1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 19, 20, 21, 22, 23, 25, 26, 27, 28, 29])

        radius = 4
        self.assertEqual(self.grid_four_equal_width_height.get_neighborhood(point, radius), [15])

    def test_neighbour_grid_equal_width_height_big_radius(self):
        '''
        Ensure gird gives accurate results for neighbouring
        grids with fixed raidus and fixed point
        '''
        point = (11, 11)
        radius = 17
        self.assertEqual(self.grid_four_equal_width_height.get_neighborhood(point, radius),
        [8, 9, 10, 11, 12, 14, 15, 16, 17, 18, 20, 21, 22, 23, 24, 26, 27, 28, 29, 30,
        32, 33, 34, 35, 36])


    def test_neighbour_grid_equal_width_height_right_horizontal(self):
        '''
        Ensure gird gives accurate results for neighbouring
        grids with fixed raidus and fixed point for right horizontal end point
        '''
        point = (10, 6)
        radius = 6
        self.assertEqual(self.grid_four_equal_width_height.get_neighborhood(point, radius), [35, 36, 23, 24, 29, 30])
    
    def test_neighbour_grid_equal_width_height_upper_verticle(self):
        '''
        Ensure gird gives accurate results for neighbouring
        grids with fixed raidus and fixed point for upper verticle end point
        '''
        point = (-2, 10)
        radius = 6
        self.assertEqual(self.grid_four_equal_width_height.get_neighborhood(point, radius), [32, 33, 34, 26, 27, 28])

    def test_neighbour_grid_equal_width_height_lower_verticle(self):
        '''
        Ensure gird gives accurate results for neighbouring
        grids with fixed raidus and fixed point for lower verticle end point
        '''
        point = (6, -10)
        radius = 6
        self.assertEqual(self.grid_four_equal_width_height.get_neighborhood(point, radius), [4, 5, 6, 10, 11, 12])

    def test_neighbour_grid_equal_width_height_left_horizontal(self):
        '''
        Ensure gird gives accurate results for neighbouring
        grids with fixed raidus and fixed point for left horizontal end point
        '''
        point = (-10, 2)
        radius = 6
        self.assertEqual(self.grid_four_equal_width_height.get_neighborhood(point, radius), [13, 14, 19, 20, 25, 26])

    def test_neighbour_grid_equal_width_height_bottom_left(self):
        '''
        Ensure gird gives accurate results for neighbouring
        grids with fixed raidus and fixed point for bottom left end point
        '''
        point = (-10, -10)
        radius = 6
        self.assertEqual(self.grid_four_equal_width_height.get_neighborhood(point, radius), [8, 1, 2, 7])

    def test_neighbour_grid_equal_width_height_top_left(self):
        '''
        Ensure gird gives accurate results for neighbouring
        grids with fixed raidus and fixed point for top left end point
        '''
        point = (-10, 10)
        radius = 6
        self.assertEqual(self.grid_four_equal_width_height.get_neighborhood(point, radius), [32, 25, 26, 31])


    def test_neighbour_grid_equal_width_height_bottom_right(self):
        '''
        Ensure gird gives accurate results for neighbouring
        grids with fixed raidus and fixed point for bottom right end point
        '''
        point = (10, -10)
        radius = 6
        self.assertEqual(self.grid_four_equal_width_height.get_neighborhood(point, radius), [11, 12, 5, 6])

    def test_neighbour_grid_equal_width_height_top_right(self):
        '''
        Ensure gird gives accurate results for neighbouring
        grids with fixed raidus and fixed point for bottom right end point
        '''
        point = (10, 10)
        radius = 6
        self.assertEqual(self.grid_four_equal_width_height.get_neighborhood(point, radius), [35, 36, 29, 30])                        

    def test_neighbour_grid_unequal_width_height(self):
        '''
        Ensure gird gives accurate results for neighbouring
        grids with fixed raidus and fixed point
        '''
        point = (-2, -2)

        radius = 4
        self.assertEqual(self.grid_four_unequal_width_height.get_neighborhood(point, radius), [9])

        radius = 5
        self.assertEqual(self.grid_four_unequal_width_height.get_neighborhood(point, radius), [2, 3, 4, 8, 9, 10, 14, 15, 16])

    def test_neighbour_grid_unequal_width_height_big_radius(self):
        '''
        Ensure gird gives accurate results for neighbouring
        grids with fixed raidus and fixed point
        '''
        point = (11, 7)

        radius = 17
        self.assertEqual(self.grid_four_unequal_width_height.get_neighborhood(point, radius),
        [2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 14, 15, 16, 17, 18, 20, 21, 22, 23, 24])

    def test_neighbour_grid_five(self):
        '''
        Ensure gird gives accurate results for neighbouring
        grids with fixed raidus and fixed point
        '''
        point = (2, -2)

        radius = 5
        self.assertEqual(self.grid_five.get_neighborhood(point, radius), [7])

        radius = 6
        self.assertEqual(self.grid_five.get_neighborhood(point, radius), [2, 3, 4, 6, 7, 8, 10, 11, 12])
