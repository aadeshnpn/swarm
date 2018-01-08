from swarms.model import EnvironmentModel

# Global variables for width and height
width = 100
height = 100

def main():
    

    env = EnvironmentModel(100, width, height, 10, 123)

    for i in range(1000):
        env.step()

    for agent in env.schedule.agents:
        print(agent.name, agent.wealth)


if __name__ == '__main__':
    main()
    