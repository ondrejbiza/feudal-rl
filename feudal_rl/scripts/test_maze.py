import numpy as np
import matplotlib.pyplot as plt
from ..maze import Maze

size = 32
maze = Maze(32, multiplex_state=False)

while True:

    x, y = maze.get_state()

    img = np.zeros((size, size), dtype=np.float32)
    img[x, y] = 1.0
    img[maze.goal_x, maze.goal_y] = 2.0

    plt.imshow(img)
    plt.show()

    action = None

    while action is None:
        action_text = input("Action: ")

        if action_text == "w":
            action = Maze.A_UP
        elif action_text == "s":
            action = Maze.A_DOWN
        elif action_text == "a":
            action = Maze.A_LEFT
        elif action_text == "d":
            action = Maze.A_RIGHT

    print("action: {:d}".format(action))
    _, done = maze.step(action)

    if done:
        maze.reset()
