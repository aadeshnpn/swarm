from swarms.model import EnvironmentModel

# Global variables for width and height
width = 100
height = 100


def main():

    env = EnvironmentModel(1000, width, height, 10, 123)

    for i in range(1000):
        env.step()

    max_wealth = 0
    max_agent = 0
    for agent in env.schedule.agents:
        if agent.wealth > max_wealth:
            max_wealth = agent.wealth
            max_agent = agent.name
    print(max_agent, max_wealth)


if __name__ == '__main__':
    main()
