import random

class Environment:
    def __init__(self, config):
        self.config = config
        self.state = None

    def reset(self):
        self.state = random.choice(list(self.config.transition_probs.keys()))
        return self.state

    def step(self, action):
        probs = self.config.transition_probs[self.state][action]
        next_state = random.choices(list(probs.keys()), weights=list(probs.values()))[0]
        reward = self.config.utility_map.get(next_state, 0)
        done = next_state in self.config.end_states
        self.state = next_state
        return next_state, reward, done

    def get_true_transition_probs(self, state, action):
        return self.config.transition_probs[state][action]