""" A general implementation of game play in open gym which uses Temporal Difference"""

import gym
import numpy as np

np.random.seed(10)
from gym.envs.classic_control import MountainCarEnv

game_env = MountainCarEnv()

game = "CartPole-v0"
# game = 'MountainCar-v0'
# game = 'MountainCarContinuous-v0'
env = gym.make(game)

sp = env.action_space
print(list(sp))

def discrete(state, round=2):
    """ Discretize the states """
    return tuple(np.round(s, round) for s in state)


policy = dict()
value = {}
learning_step = 0.1
discount = 1


def argmax_action(input_state):
    """ Find the best action for a particular state """

    action_space = [0, 1, 2]  # Change this line to make things more general
    max_value, optimal_action = -1, env.action_space.sample()  # Do random action

    # Take all the possible actions from given input state
    for action in action_space:
        game_env.state = input_state  # Fix the state
        state, reward, done, info = game_env.step(action)
        state = discrete(state)  # This could also be a never seen before state

        temp_value = value.get(state, -1)
        if temp_value == -1:
            value[state] = 0  # If the state is new, register it into value dict

        if temp_value > max_value:
            max_value = temp_value
            optimal_action = action

    return optimal_action


def play_game():
    """ Play one episode of the game, update policy and value for states """

    episode_states = []
    state = env.reset()  # Start the game

    state = discrete(state)
    policy[state] = env.action_space.sample()
    value[state] = 0

    episode_states.append(state)
    done = False  # Is the episode complete?

    while not done:

        old_state = state
        state, reward, done, info = env.step(policy.get(state, env.action_space.sample()))

        state = discrete(state)
        episode_states.append(state)

        # If this new state does not exist, register it into the dictionary
        if value.get(state, -1) == -1:
            value[state] = 0
            policy[state] = env.action_space.sample()
            # print("New random action warranted!", state, policy[state], policy.items().__len__())
        else:
            # print("Using the action from policy ", state, policy[state], policy.items().__len__())
            pass

        # TD(O) for estimating v_policy
        value[old_state] = value[old_state] + learning_step * (reward + discount * value[state] - value[old_state])


        # if reward != -1:
        #     print("Yay, cracked!", reward)

        # env.render()
    # env.close()

    # Update the policy for each of the states encountered in the episode
    for index, state in enumerate(episode_states):
        # old_action = policy.get(state, env.action_space.sample())  # This should be issue for only the last ep state
        policy[state] = argmax_action(state)
        # if old_action != policy[state]:
        #     print(index, state, " \t Updated A. ", old_action, policy[state])
        #     pass


if __name__ == '__main__':
    # for i in range(10000):
    #     play_game()
    #     if i % 10 == 0:
    #         print("Episode ", i, "States : ", policy.items().__len__())
    # pass

    print(value)
    print(policy)
    pass