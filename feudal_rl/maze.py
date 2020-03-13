import numpy as np


class Maze:

    A_UP = 0
    A_DOWN = 1
    A_LEFT = 2
    A_RIGHT = 3

    def __init__(self, size, multiplex_state=True):

        self.size = size
        self.multiplex_state = multiplex_state

        self.pos_x, self.pos_y, self.goal_x, self.goal_y = None, None, None, None

        self.set_random_goal()
        self.reset()

    def reset(self):

        pos = np.random.randint(0, self.size, size=2)
        self.pos_x = pos[0]
        self.pos_y = pos[1]

        while self.pos_x == self.goal_x and self.pos_y == self.goal_y:
            pos = np.random.randint(0, self.size, size=2)
            self.pos_x = pos[0]
            self.pos_y = pos[1]

    def step(self, action):

        if action == self.A_UP:
            if self.pos_x > 0:
                self.pos_x -= 1
        elif action == self.A_DOWN:
            if self.pos_x < self.size - 1:
                self.pos_x += 1
        elif action == self.A_LEFT:
            if self.pos_y > 0:
                self.pos_y -= 1
        elif action == self.A_RIGHT:
            if self.pos_y < self.size - 1:
                self.pos_y += 1
        else:
            raise ValueError("Invalid action.")

        return self.get_state(), self.check_goal()

    def get_state(self):

        if self.multiplex_state:
            return self.pos_x * self.size + self.pos_y
        else:
            return self.pos_x, self.pos_y

    def check_goal(self):

        return self.pos_x == self.goal_x and self.pos_y == self.goal_y

    def set_random_goal(self):

        goal = np.random.randint(0, self.size, size=2)
        self.goal_x = goal[0]
        self.goal_y = goal[1]
