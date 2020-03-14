import numpy as np


class FeudalAgent:

    NUM_TOP_ACTIONS = 5
    NUM_BOTTOM_ACTIONS = 4

    def __init__(self, size, agents_per_level, make_policy, alpha, gamma):

        self.size = size
        self.hierarchy = []
        self.cells_per_agent = []

        for level_idx, num_agents in enumerate(reversed(agents_per_level)):

            if level_idx == 0:
                num_cells = size ** 2
                num_actions = self.NUM_BOTTOM_ACTIONS
            else:
                num_cells = len(self.hierarchy[-1])
                num_actions = self.NUM_TOP_ACTIONS

            if level_idx == len(agents_per_level) - 1:
                num_master_actions = 1
            else:
                num_master_actions = self.NUM_TOP_ACTIONS

            assert num_cells % num_agents == 0
            cells_per_agent = num_cells // num_agents
            self.hierarchy.append([
                CellAgent(cells_per_agent, num_actions, num_master_actions, make_policy(num_actions), alpha, gamma)
                for _ in range(num_agents)
            ])
            self.cells_per_agent.append(cells_per_agent)

        self.current_level = len(self.hierarchy) - 1

    def act(self, x, y, master_action=None):

        print("act level:", self.current_level)

        if self.current_level == 0:
            cell = self.get_cell(self.current_level, x, y)
            if master_action is not None:
                cell.master_action = master_action
            action = cell.act(self.multiplex_state_for_cell(self.current_level, x, y))
            return action
        elif master_action is not None:
            cell = self.get_cell(self.current_level, x, y)
            if master_action is not None:
                cell.master_action = master_action
            action = cell.act(self.multiplex_state_for_cell(self.current_level, x, y))
            self.current_level -= 1
            return self.act(x, y, master_action=action)
        else:
            cell = self.get_cell(self.current_level, x, y)
            cell.master_action = 0
            action = cell.act(self.multiplex_state_for_cell(self.current_level, x, y))
            self.current_level -= 1
            return self.act(x, y, master_action=action)

    def backup(self, current_x, current_y, next_x, next_y, action, reward):

        print("backup level:", self.current_level)

        cell = self.get_cell(self.current_level, current_x, current_y)
        master_action = cell.master_action

        current_cell_x, current_cell_y = self.get_cell_x_y(self.current_level + 1, current_x, current_y)
        next_cell_x, next_cell_y = self.get_cell_x_y(self.current_level + 1, next_x, next_y)

        if current_cell_x != next_cell_x or current_cell_y != next_cell_y:

            #if master_action == 4:
            #    meta_reward = int(reward > 0)

            self.current_level = self.current_level + 1
            self.backup(current_x, current_y, next_x, next_y, action, reward)

    def get_cell_x_y(self, level, x, y):

        length = int(np.sqrt(len(self.hierarchy[level])))
        length_per_cell = self.size // length

        level_x = x // length_per_cell
        level_y = y // length_per_cell

        return level_x, level_y

    def get_cell(self, level, x, y):

        length = int(np.sqrt(len(self.hierarchy[level])))
        length_per_cell = self.size // length

        level_x = x // length_per_cell
        level_y = y // length_per_cell

        idx = level_x * length + level_y
        return self.hierarchy[level][idx]

    def multiplex_state_for_cell(self, level, x, y):

        length = int(np.sqrt(len(self.hierarchy[level])))

        if level == 0:
            length_below = self.size
        else:
            length_below = int(np.sqrt(len(self.hierarchy[level - 1])))

        length_per_cell = length_below // length

        level_x = x // length_per_cell
        level_y = y // length_per_cell

        cell_x = x - level_x * length_per_cell
        cell_y = y - level_y * length_per_cell

        return cell_x * length_per_cell + cell_y


class CellAgent:

    def __init__(self, num_states, num_actions, num_master_actions, policy, alpha, gamma):

        self.qs = np.zeros((num_actions, num_master_actions, num_states))
        self.policy = policy
        self.alpha = alpha
        self.gamma = gamma

        self.master_action = None

    def act(self, state):

        qs = self.qs[:, self.master_action, state]
        return self.policy.act(qs)

    def learn(self, state, action, reward, next_state, done):

        if done:
            self.qs[action, self.master_action, state] += self.alpha * (
                reward - self.qs[action, self.master_action, state]
            )
        else:
            next_qs = self.qs[:, self.master_action, next_state]
            target = reward + self.gamma * np.max(next_qs)

            self.qs[action, self.master_action, state] += self.alpha * (
                target - self.qs[action, self.master_action, state]
            )

    def order(self, master_action):

        self.master_action = master_action
