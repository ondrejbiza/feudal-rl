import argparse
import numpy as np
import matplotlib.pyplot as plt
from ..maze import Maze
from ..policy import PolicyEpsGreedy, PolicySoftmax
from ..q_agent import QAgent


def main(args):

    maze = Maze(args.size)
    policy = PolicySoftmax(args.tau, 4)
    agent = QAgent(args.size ** 2, policy, args.alpha, args.gamma)

    steps = 0
    for i in range(args.num_steps):

        if i > 0 and i % 50000 == 0:
            print("Learning step {:d}".format(i))

            qs = agent.qs
            qs = np.reshape(qs, (4, args.size, args.size))
            plt.imshow(qs[0])
            plt.colorbar()
            plt.show()

        state = maze.get_state()
        action = agent.act(state)
        next_state, done = maze.step(action)
        agent.learn(state, action, int(done), next_state)
        steps += 1

        if done:
            maze.reset()
            print("Goal reached in {:d} steps".format(steps))
            steps = 0


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--size", type=int, default=32)
    parser.add_argument("--alpha", type=float, default=0.01)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=1.0)
    parser.add_argument("--num-steps", type=int, default=200000)

    parsed = parser.parse_args()
    main(parsed)
