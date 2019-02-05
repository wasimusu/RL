import gym
import numpy as np

policy = dict()
Q = {}
alpha = 0.1
discount = 0.9
epsilon = 0.05  # Percentage of random exploration

# game = "CartPole-v0"
game = 'MountainCar-v0'
env = gym.make(game)
action_space = [0, 1, 2]  # Change this line to make things more general


def discrete(state, weights=(1, 2)):
    """ Discretize the states """
    return tuple(np.round(attribute, weights[index]) for index, attribute in enumerate(state))


def argmax_action(state):
    max_Q = -1
    optimal_action = env.action_space.sample()
    for action in action_space:
        temp_Q = Q.get((state, action), -1)
        if temp_Q == -1:
            Q[(state, action)] = 0

        if temp_Q > max_Q:
            optimal_action = action
            temp_Q = max_Q
    pass


if __name__ == '__main__':
    state = (1.123, 1.12454578)
    state = discrete(state)
    print(state)
