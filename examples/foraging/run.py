"""Main script to run."""

from model import SingleCarryDropReturnSwarmEnvironmentModel

width = 1000
height = 800


def main():
    """Create the environment.

    Custom environment is created and experiment is ran.
    """
    env = SingleCarryDropReturnSwarmEnvironmentModel(
        100, width, height, 10, 123, True)
    for i in range(2500):
        # print (env.agents[0].location)
        env.step()

    grid = env.grid
    food_loc = (0, 0)
    neighbours = grid.get_neighborhood(food_loc, 5)
    food_objects = grid.get_objects_from_list_of_grid('Food', neighbours)
    print (len(food_objects))
    #for food in food_objects:
    #    print('food', food.id, food.location)

if __name__ == '__main__':
    main()
