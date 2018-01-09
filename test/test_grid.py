'''
Test the grid object
'''
import unittest

from lib.space import Grid

"""
Simple grid for testing
Widht:20, Height:20, Grid:5
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
        self.assertEqual(self.grid_four_equal_width_height.find_grid((-1, 1))[1], 21)
        self.assertEqual(self.grid_four_equal_width_height.find_grid((4, 4))[1], 29)

    def test_location_grid_four_unequal_width_height(self):
        '''
        For grid 24x16, point (-5,1) should lie in grid number 8
        '''
        self.assertEqual(self.grid_four_unequal_width_height.find_grid((-5, 1))[1], 14)
        self.assertEqual(self.grid_four_unequal_width_height.find_grid((7, 7))[1], 23)

    def test_location_grid_five(self):
        '''
        For grid 20x20, point (-1,4) should lie in grid number 10
        '''        
        self.assertEqual(self.grid_five.find_grid((-1, 4))[1], 10)
        self.assertEqual(self.grid_five.find_grid((9, 9))[1], 16)
        self.assertEqual(self.grid_five.find_grid((-4, -4))[1], 6)
        self.assertEqual(self.grid_five.find_grid((-9, 3))[1], 9)

    def test_neighboring_grids_with_radius_ten(self):
        '''
        Ensure gird gives accurate results for neighbouring
        grids with fixed raidus and fixed point
        '''
        #point = (-1, 1)
        #radius = 5
        #neighbours = self.grid.get_neighborhood(point, radius)
        #print(neighbours)
        pass


if __name__ == '__main__':
    unittest.main()
