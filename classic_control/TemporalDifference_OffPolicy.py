""" A general implementation of game play in open gym which uses Temporal Difference"""

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
    """ Find the best action for a particular state """

    max_Q, optimal_action = -1, env.action_space.sample()  # Assign random action as optimal action

    # Take all the possible actions from given input state
    for action in action_space:
        temp_Q = Q.get((state, action), -1)

        if temp_Q == -1:  # Q(S, A) may not exist (even for an existing state)
            # If the state is new, register it into Q(S, A) dict with random value
            Q[(state, action)] = np.random.uniform(0.2, 0.5)
            Q[(state, action)] = 0.5

        if temp_Q > max_Q:
            max_Q = temp_Q
            optimal_action = action

    return optimal_action


def update_Q(state, action, new_state, reward):
    """ Update Q using off policy TD control """

    # This does not exist for newly observed states
    if Q.get((state, action), -1) == -1:
        # Q[(state, action)] = np.random.uniform(0, 0.5)
        Q[(state, action)] = 0.5

    Q[(state, action)] = Q[(state, action)] + alpha * (reward + discount * Q[(new_state, argmax_action(new_state))]
                                                       - Q[state, action])


def play_game(episode_count):
    """ Play one episode of the game, update policy and value for states """

    episode_states = []
    new_state = env.reset()  # (Re)Start the game

    new_state = discrete(new_state)
    policy[new_state] = env.action_space.sample()
    episode_states.append(new_state)

    terminal_state = False  # Is the episode complete?
    reward_ep = 0
    while not terminal_state:
        # epsilon-greedy exploration and exploitation
        if np.random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()
        else:
            # A registered state always have an associated action
            # It should never raise KeyError
            action = policy.get(new_state)

        state = new_state
        # Observe and log newly observed state
        new_state, reward, terminal_state, info = env.step(action)
        new_state = discrete(new_state)
        episode_states.append(new_state)

        reward_ep += reward  # Total sum in an episode

        # If this is a brand new state it does not have associated action
        if policy.get(new_state, -1) == -1:
            policy[new_state] = env.action_space.sample()

        # if terminal_state:  # If this is terminal state
        #     # If these terminal states has not been seen before, assign Q(S, A) = 0
        #     for action in action_space:
        #         if Q.get((new_state, action), -1) == -1:
        #             Q[(new_state, action)] = 1
        #             # Q[(state, action)] = 0

        # Q - learning for estimating optimal policy
        update_Q(state, action, new_state, reward)

        if episode_count >= 2000:
            env.render()

    if episode_count >= 2000:
        env.close()

    # Update the policy for each of the states encountered in the episode
    for state in episode_states:
        policy[state] = argmax_action(state)


if __name__ == '__main__':
    for i in range(10000):
        play_game(i)
        if i % 100 == 0:
            print("Episode ", i, "States : ", policy.items().__len__())
            pass

    # TODO: Every nth episode check if Q is converging or not