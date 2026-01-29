import numpy as np
from utils import one_hot_featurize, random_featurize

class GarnetEnv:
    def __init__(self, num_states, num_actions, branching_factor,
                 featurize=one_hot_featurize, size_feature=None,
                 generation_seed=None, seed=None):
        self.num_states = num_states
        self.num_actions = num_actions
        self.branching_factor = branching_factor
        self.generation_rng = np.random.default_rng(generation_seed)
        self.rng = np.random.default_rng(seed)

        self.transitions = np.array([ [ self.generation_rng.choice(num_states, branching_factor, replace=False)
                                        for a in range(num_actions) ]
                                      for s in range(num_states) ])

        
        self.probs = self.generation_rng.uniform(size=(num_states, num_actions, branching_factor))
        self.probs /= self.probs.sum(axis=2, keepdims=True)

        
        self.rewards = self.generation_rng.uniform(low=0, high=1, size=(num_states, num_actions))

        self.state = self.generation_rng.integers(0, num_states)

        size_feature = num_states * num_actions if size_feature is None else size_feature
        self.features = np.array([[ featurize(s, a, size_feature,
                                              self.generation_rng, num_actions)
                                    for a in range(num_actions) ]
                                  for s in range(num_states)
                                  ])

    def reset(self):
        self.state = self.rng.integers(0, self.num_states)
        return self.state

    def step(self, action):
        assert 0 <= action < self.num_actions
        s = self.state
        next_states = self.transitions[s][action]
        probs = self.probs[s][action]
        next_state = self.rng.choice(next_states, p=probs)
        reward = self.rewards[s][action]
        self.state = next_state
        return next_state, reward
