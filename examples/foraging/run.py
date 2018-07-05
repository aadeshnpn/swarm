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
    for i in range(100000):
        env.step()


if __name__ == '__main__':
    main()
