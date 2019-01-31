""" A general implementation of game play in open gym which uses Temporal Difference"""

import gym
import numpy as np


def discrete(state, round=1):
    """ Discretize the states """
    return tuple(np.round(s, round) for s in state)


policy = dict()
value = {}
learning_step = 0.1
discount = 1


def argmax_action(state):
    """ Find the best action for a particular state """
    # Can you reset the environment to a particular state ?
    pass


def play_game():
    game = "CartPole-v0"
    game = 'MountainCar-v0'
    env = gym.make(game)

    # Play one episode of the game
    episode_states = []
    state = env.reset()  # Start the game
    state = discrete(state)
    print("State : ", state)
    episode_states.append(state)
    done = False  # Is the episode complete?
    while not done:
        old_state = state

        # If you have encountered the state, play according to policy
        # Else build a random policy for it
        action = policy.get(state, -1)
        # If the state was not defined
        if action == -1:
            policy[state] = env.action_space.sample()
            action = policy.get(state, -1)
            value[state] = 0

        state, reward, done, info = env.step(action)
        state = discrete(state)

        # If the new state does not exist, register it into the dictionary
        if value.get(state, -1) == -1:
            value[state] = 0

        # TD(O) for estimating v_policy
        value[old_state] = value[old_state] + learning_step * (reward + discount * value[state] - value[state])
        episode_states.append(state)

        print("State : ", state, "\tReward : ", reward)
        env.render()

    for state in episode_states:
        policy[state] = argmax_action(state)

    env.close()


if __name__ == '__main__':
    play_game()
