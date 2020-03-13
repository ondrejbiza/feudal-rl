import numpy as np


class QAgent:

    NUM_ACTIONS = 4

    def __init__(self, num_states, policy, alpha, gamma):

        self.qs = np.zeros((self.NUM_ACTIONS, num_states))
        self.policy = policy
        self.alpha = alpha
        self.gamma = gamma

    def act(self, state):

        qs = self.qs[:, state]
        return self.policy.act(qs)

    def learn(self, state, action, reward, next_state):

        next_qs = self.qs[:, next_state]
        target = reward + self.gamma * np.max(next_qs)

        self.qs[action, state] += self.alpha * (target - self.qs[action, state])
