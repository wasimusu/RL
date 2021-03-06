""" Python implementation of Monte Carlo Simulation on BlackJack using Gym"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import gym

env = gym.make('Blackjack-v0')


def argmax_action(state):
    """
    Find the best action which leads to the highest Q for a given state.

    Algorithm:
        At each state I can only take two actions : HIT / STICK
        Return the action having higher Q
     """
    q_stick = Q[state]

    deck = [2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10, 11]
    drawn_card = np.random.choice(deck)
    player_sum, upcard, useable_ace = state
    if drawn_card == 1:
        useable_ace = True

    player_sum = player_sum + drawn_card
    if useable_ace == True:
        player_sum -= 10

    q_hit = Q[state_to_string(player_sum, upcard, useable_ace)]
    if q_hit > q_stick:
        return HIT
    else:
        return STICK


def state_to_string(player_sum, upcard=2, useable_ace=True):
    """
    Convert game stat in state
    :param upcard : the card that the dealer is showing
    """

    # Considering all the states below 12 as 11
    if player_sum < 12:
        player_sum = 11
        upcard = 2
        useable_ace = False

    # Considering all the states above 21 as 22
    if player_sum > 21:
        player_sum = 22
        upcard = 2
        useable_ace = False

    state = (player_sum, upcard, useable_ace)
    return state


HIT, STICK = True, False

# 12-21 * USEABLE_ACE * DEALER 2-11 : 10 * 2 * 10
states = []
states_action = []
for player_sum in range(12, 22):
    # 0 - no useable ace. 1 - useable ace
    for useable_ace in [True, False]:
        for upcard in range(1, 11):
            state = state_to_string(player_sum, upcard, useable_ace)
            states.append(state)

state_11 = state_to_string(11)
state_22 = state_to_string(22)

policy = dict(zip(states, np.random.uniform(0, 1, 200)))  # 400 = Number of States * Number of Action
policy[state_11] = 0  # policy for state less than 12
policy[state_22] = 0  # policy for state greater than 21

Q = dict(zip(states, np.random.uniform(0, 1, 200)))  # 400 = Number of States * Number of Action
Q[state_11] = 0  # Q for state less than 12
Q[state_22] = 0  # Q for state greater than 21

state_visit_count = dict(zip(states, [0] * len(states)))  # Keeping count of the visit for each state
state_visit_count[state_22] = 0  # policy for state greater than 21
state_visit_count[state_11] = 0  # policy for state less than 12


def updateQ(state, reward):
    """ Apply incremental update to Q values for given state """
    player_sum, upcard, useable_ace = state
    state = state_to_string(player_sum, upcard, useable_ace)
    state_visit_count[state] += 1
    Q[state] = Q[state] + (1.0 / state_visit_count[state]) * (reward - Q[state])


def play_blackjack():
    # Draw cards until the player's hand sum to 12

    episodic_states = []  # Track states in an episode
    actions = []  # Series of actions that were taken

    player_sum, upcard, useable_ace = env.reset()  # Re-start the game
    episodic_states.append((player_sum, upcard, useable_ace))

    while True:
        # Play according to policy
        if player_sum < 20:  # Draw card
            state, reward, game_over, _ = env.step(True)
            episodic_states.append(state)
            player_sum = state[0]
            actions.append(HIT)

        else:
            # Let the game / dealer finish
            # Step until the game is complete
            game_over = False
            while not game_over:
                state, reward, game_over, _ = env.step(False)
                episodic_states.append(state)
                actions.append(STICK)

            # Determine the series of actions
            for act, ep_state in zip(actions, episodic_states):
                updateQ(ep_state, reward)
            break


def evaluate_policy():
    for state in states:
        policy[state] = argmax_action(state)


if __name__ == '__main__':

    for iteration in range(1000001):
        play_blackjack()

        if iteration == 10000 or iteration == 1000000:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')

            x, y = [], []
            z_ace = []
            z_no_ace = []
            for ps in range(12, 22):
                for upcard in range(1, 11):
                    x.append(ps)
                    y.append(upcard)
                    z_ace.append(Q[state_to_string(ps, upcard, True)])
                    z_no_ace.append(Q[state_to_string(ps, upcard, False)])

            ax = fig.gca(projection='3d')
            ax.set_xlabel("Players Hand")
            ax.set_ylabel("Upcard")
            ax.set_zlabel("Q Value")

            # # Useable ace
            # ax.plot_trisurf(x, y, z_ace, cmap=plt.cm.viridis, linewidth=0.2)
            # plt.title("Useable Ace {} Iteration".format(iteration))

            # Unuseable ace
            ax.plot_trisurf(x, y, z_no_ace, cmap=plt.cm.viridis, linewidth=0.2)
            plt.title("No Useable Ace - {} Iteration".format(iteration))

            plt.show()
            plt.pause(1)

    # evaluate_policy()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    x, y = [], []
    z_ace = []
    z_no_ace = []
    for ps in range(12, 22):
        for upcard in range(1, 11):
            x.append(ps)
            y.append(upcard)
            z_ace.append(policy[state_to_string(ps, upcard, True)])
            z_no_ace.append(policy[state_to_string(ps, upcard, False)])

    ax = fig.gca(projection='3d')
    ax.set_xlabel("Players Hand")
    ax.set_ylabel("Upcard")
    ax.set_zlabel("HIT / STICK")

    # # Useable ace
    # ax.plot_trisurf(x, y, z_ace, cmap=plt.cm.viridis, linewidth=0.2)
    # plt.title("Useable Ace {} Iteration".format(iteration))

    ax.plot_trisurf(x, y, z_no_ace, cmap=plt.cm.viridis, linewidth=0.2)
    plt.title("No Useable Ace {} Iteration".format(iteration))

    plt.show()
    plt.pause(1)
