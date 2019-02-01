""" A general implementation of game play in open gym which uses Temporal Difference"""

import gym
import numpy as np

np.random.seed(10)
from gym.envs.classic_control import MountainCarEnv

game_env = MountainCarEnv()

# game = "CartPole-v0"
game = 'MountainCar-v0'
# game = 'MountainCarContinuous-v0'
env = gym.make(game)

sp = env.action_space


def discrete(state, round=3):
    """ Discretize the states """
    return tuple(np.round(s, round) for s in state)


policy = dict()
Q = {}
learning_step = 0.5
discount = 1
epsilon = 0.1  # Percentage of random exploration


def argmax_action(state):
    """ Find the best action for a particular state """

    # action_space = [0, 1, 2]  # Change this line to make things more general
    action_space = [0, 1, 2]  # Change this line to make things more general
    max_Q, optimal_action = -1, env.action_space.sample()  # Do random action

    # Take all the possible actions from given input state
    for action in action_space:
        temp_Q = Q.get((state, action), -1)

        if temp_Q == -1:  # Q(S, A) may not exist (even for an existing state)
            # If the state is new, register it into Q(S, A) dict with random value
            Q[(state, action)] = np.random.uniform(0, 1)
            Q[(state, action)] = 1

        if temp_Q > max_Q:
            max_Q = temp_Q
            optimal_action = action

    return optimal_action


def update_Q(old_state, action, new_state, reward):
    """ Update Q using off policy TD control """
    # print("Old state : {} \tAction : {} \tNew state : {} \tReward : {}".format(old_state, action, new_state, reward))

    # This does not exist for the first iteration
    if Q.get((old_state, action), -1) == -1:
        Q[(old_state, action)] = np.random.uniform(0, 0.05)

    Q[(old_state, action)] = Q[(old_state, action)] + learning_step * (reward + discount *
                                                                       Q[(new_state, argmax_action(new_state))] - Q[
                                                                           old_state, action])


def play_game():
    """ Play one episode of the game, update policy and value for states """

    episode_states = []
    state = env.reset()  # Start the game

    state = discrete(state)
    policy[state] = env.action_space.sample()
    Q[(state, env.action_space.sample())] = 0
    episode_states.append(state)

    # TODO : Q[terminal, action] = 0 ? What is the terminal state

    done = False  # Is the episode complete?
    reward_ep = 0
    while not done:
        # epsilon-greedy exploration and exploitation
        if np.random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()
        else:
            # If the current policy has associated action for given state choose it
            # A registered state always have an associated action
            # It should never raise KeyError
            action = policy.get(state)

        old_state = state
        state, reward, done, info = env.step(action)
        reward_ep += reward

        state = discrete(state)
        episode_states.append(state)

        # If this is a brand new state it does not have associated action
        if policy.get(state, -1) == -1:
            policy[state] = env.action_space.sample()

        if done:  # If this is terminal state
            # If these terminal states has not been seen before, assign Q(S, A) = 0
            for action in [0, 1, 2]:  # Make this line general
                if Q.get((state, action), -1) == -1:
                    Q[(state, action)] = 0

        # Q - learning for estimating optimal policy
        update_Q(old_state, action, state, reward)

        env.render()
    env.close()

    print("Episode reward : ", reward_ep)
    # Update the policy for each of the states encountered in the episode
    for index, state in enumerate(episode_states):
        policy[state] = argmax_action(state)


if __name__ == '__main__':
    for i in range(10000):
        play_game()
        if i % 10 == 0:
            # print("Episode ", i, "States : ", policy.items().__len__())
            pass
    pass
