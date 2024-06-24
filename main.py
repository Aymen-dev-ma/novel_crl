from agent import Agent
from environment import Environment
from config import Config

def main():
    config = Config()
    env = Environment(config)
    agent = Agent(config)

    initial_state = env.reset()
    log = agent.play(initial_state)

    print("Simulation complete. Log:")
    for entry in log:
        print(f"Step {entry[0]}: State={entry[1]}, Action={entry[2]}, Utility={entry[3]}")

    # Additional analysis
    reactive_actions = [agent.react(entry[1], entry[0]).item() for entry in log]
    deliberative_actions = [entry[2] for entry in log]

    print("\nAction Analysis:")
    print(f"Reactive actions: {reactive_actions}")
    print(f"Deliberative actions: {deliberative_actions}")
    print(f"Number of times deliberative overrode reactive: {sum(r != d for r, d in zip(reactive_actions, deliberative_actions))}")

if __name__ == "__main__":
    main()