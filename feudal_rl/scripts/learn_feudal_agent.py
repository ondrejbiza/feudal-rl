import argparse
import numpy as np
import matplotlib.pyplot as plt
from ..maze import Maze
from ..feudal_agent import FeudalAgent
from ..policy import PolicySoftmax


def make_policy(num_actions):
    return PolicySoftmax(0.1, num_actions)


def main(args):

    maze = Maze(args.size, multiplex_state=False)
    agent = FeudalAgent(args.size, [4, 16, 64], make_policy, args.alpha, args.gamma)

    steps = 0
    while True:

        x, y = maze.get_state()

        img = np.zeros((args.size, args.size), dtype=np.float32)
        img[x, y] = 1.0
        img[maze.goal_x, maze.goal_y] = 2.0

        plt.imshow(img)
        plt.show()

        action = agent.act(x, y)
        next_state, done = maze.step(action)
        print(action)
        #agent.learn(state, action, int(done), next_state)
        steps += 1

        agent.backup(x, y, next_state[0], next_state[1], action, int(done))

        if done:
            maze.reset()
            print("Goal reached in {:d} steps".format(steps))
            steps = 0


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--size", type=int, default=32)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.1)
    parser.add_argument("--num-steps", type=int, default=200000)

    parsed = parser.parse_args()
    main(parsed)
