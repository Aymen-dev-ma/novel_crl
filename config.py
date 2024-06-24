class Config:
    def __init__(self):
        self.num_actions = 2
        self.max_steps = 100
        self.target_utility = 10.0  # Target utility for conditioning
        self.end_states = {3}
        self.utility_map = {0: -1, 1: 0, 2: 1, 3: 10}
        self.transition_probs = {
            0: {0: {0: 0.7, 1: 0.3}, 1: {1: 0.8, 2: 0.2}},
            1: {0: {0: 0.1, 1: 0.9}, 1: {1: 0.5, 2: 0.5}},
            2: {0: {1: 0.4, 2: 0.4, 3: 0.2}, 1: {2: 0.3, 3: 0.7}},
            3: {0: {3: 1.0}, 1: {3: 1.0}}
        }