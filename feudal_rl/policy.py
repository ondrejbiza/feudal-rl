import numpy as np
from scipy.special import softmax


class PolicyEpsGreedy:

    def __init__(self, eps, num_actions):
        self.eps = eps
        self.num_actions = num_actions

    def act(self, q_values):
        if np.random.uniform(0, 1) < self.eps:
            return np.random.randint(0, self.num_actions)
        else:
            return np.argmax(q_values)


class PolicySoftmax:

    def __init__(self, tau, num_actions):
        self.tau = tau
        self.num_actions = num_actions
        self.actions_list = list(range(self.num_actions))

    def act(self, q_values):
        probs = softmax(q_values / self.tau)
        return np.random.choice(self.actions_list, p=probs)
