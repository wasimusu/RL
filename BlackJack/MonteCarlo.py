""" Python implementation of Monte Carlo Simulation on BlackJack """

import numpy as np
import matplotlib.pyplot as plt


def state_to_string(player_sum, useable_ace=0, upcard=2):
    """
    Convert game stat in state
    :param upcard : the card that the dealer is showing
    """

    # Considering all the states below 12 as 11
    if player_sum < 12:
        player_sum = 11
        upcard = 2
        useable_ace = 0

    # Considering all the states above 21 as 22
    if player_sum > 21:
        player_sum = 22
        upcard = 2
        useable_ace = 0

    state = "PS_{}_UA_{}_UC_{}".format(player_sum, useable_ace, upcard)
    return state


def state_action_to_string(state, action):
    """ Convert game stat in state """
    return "{}_ACT_{}".format(state, action)


HIT, STICK = 1, 0
deck = [2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10, 11]

# 12-21 * USEABLE_ACE * DEALER 2-11 : 10 * 2 * 10
states = []
states_action = []
for player_sum in range(12, 21 + 1):
    # 0 - no useable ace. 1 - useable ace
    for useable_ace in [1, 0]:
        for upcard in range(2, 11 + 1):
            state = state_to_string(player_sum, useable_ace, upcard)
            states.append(state)
            for action in [HIT, STICK]:
                states_action.append(state_action_to_string(state, action))

state_11 = state_to_string(11)
state_22 = state_to_string(22)

Q = dict(zip(states_action, np.random.uniform(0, 1, 400)))  # 400 = Number of States * Number of Action
Q[state_action_to_string(state_11, 1)] = 0  # policy for state less than 12
Q[state_action_to_string(state_11, 0)] = 0  # policy for state less than 12
Q[state_action_to_string(state_22, 1)] = 0  # policy for state greater than 21
Q[state_action_to_string(state_22, 0)] = 0  # policy for state greater than 21

policy = dict(zip(states, np.random.randint(0, 1, 200)))  # 200 = One action that can be taken from each state
policy[state_22] = 0  # policy for state greater than 21
policy[state_11] = 1  # policy for state less than 12

state_visit_count = dict(zip(states, [0] * len(states)))  # Keeping count of the visit for each state
state_visit_count[state_22] = 0  # policy for state greater than 21
state_visit_count[state_11] = 0  # policy for state less than 12


def draw_card(num_cards=1):
    return int(np.random.choice(deck, num_cards))


def sum_deck(deck):
    sum_deck = sum(deck)
    if 11 in deck:
        if sum_deck > 21:
            sum_deck -= 10
    return int(sum_deck)


def test_sum_deck():
    decks = [
        [11, 4, 3, 8],
        [8, 4, 8],
        [9, 10, 11],
        [10, 10, 11],
        [2, 2, 8],
        [2, 3, 8, 8]]
    sums = [16, 20, 20, 21, 12, 21]
    for deck, sum in zip(decks, sums):
        assert sum_deck(deck) == sum


def evaluate_game(dealer_hand, player_hand):
    """
    Returns reward for by comparing dealer's and players' hand
    Returns win : 1
    loss : -1
    draw : 0
    """
    # Natural : Receiving a sum of 21 in the first two hands. A blackjack.

    sum_dealer_hand = sum_deck(dealer_hand)
    sum_player_hand = sum_deck(player_hand)

    # Both of them scoring equal or both of them being bust
    if sum_dealer_hand == sum_player_hand:
        return 0

    # Dealer bust
    if sum_dealer_hand > 21:
        return 1

    # Player bust
    if sum_player_hand > 21:
        return -1

    # Normal outcomes
    if sum_dealer_hand > sum_player_hand:
        return -1
    else:
        return 1


def updateQ(state, action, reward):
    """ Apply incremental update to Q values for given state """
    state_visit_count[state] += 1
    state_action = state_action_to_string(state, action)
    Q[state_action] = Q[state_action] + (1.0 / state_visit_count[state]) * (reward - Q[state_action])


def argmax_action(state):
    """
    Find the best action which leads to the highest Q for a given state.

    Algorithm:
        At each state I can only take two actions : HIT / STICK
        Return the action having higher Q
     """

    max_q, optimal_action = -1, -1

    # Find the action leading to the highest Q
    for action in [HIT, STICK]:
        state_action = state_action_to_string(state, action)
        if action == HIT:
            drawn_card = draw_card()
            _, PS, _, UA, _, UC = state.split("_")
            new_sum = sum_deck([int(PS), drawn_card])
            new_state = state_to_string(new_sum, UA, UC)
            state_action = state_action_to_string(new_state, action)

        if Q[state_action] > max_q:
            optimal_action = action
            max_q = Q[state_action]

    return optimal_action


def generate_states_for_player_hand(player_hand, useable_ace, upcard):
    """
    >> generate_states_for_player_hand([2, 4, 6, 8], 1, 10)
    >> ['P_2_A_1_D_10', 'P_4_A_1_D_10', 'P_7_A_1_D_10', 'P_12_A_1_D_10']
    """
    return [state_to_string(sum_deck(player_hand[:i]), useable_ace, upcard) for i in
            range(2, len(player_hand) + 1)]


def play_blackjack():
    # Draw cards until the player's hand sum to 12

    actions = []  # Series of actions that were taken

    dealer_hand, player_hand = [], []
    while True:
        dealer_hand.append(draw_card())
        player_hand.append(draw_card())
        actions.append(HIT)
        if sum_deck(player_hand) > 11: break

    upcard = dealer_hand[0]
    # If the sum if 21, the player won

    while True:
        useable_ace = 1 if 11 in player_hand else 0
        player_state = state_to_string(sum_deck(player_hand), useable_ace, upcard)

        # Play according to policy
        if policy[player_state] == HIT:  # Draw card
            dealer_hand.append(draw_card())
            player_hand.append(draw_card())
            actions.append(HIT)
        else:
            actions.append(STICK)

            # Evaluate the game for reward
            reward = evaluate_game(dealer_hand, player_hand)
            episodic_states = generate_states_for_player_hand(player_hand, useable_ace, upcard)

            assert len(actions) == len(player_hand) + 1
            actions.pop(0)
            actions.pop(0)
            assert len(actions) == len(episodic_states)

            # Determine the series of actions
            for act, ep_state in zip(actions, episodic_states):
                updateQ(ep_state, action, reward)
            break

    # Update the policy after each episode
    # Do argmax action for each state
    for i, state in enumerate(states):
        updated_action = argmax_action(state)
        if policy[state] != updated_action:
            # print("Updated : ", state, policy[state], updated_action)
            policy[state] = argmax_action(state)


if __name__ == '__main__':
    for _ in range(10000):
        print(_, sum(policy.values()))
        play_blackjack()

    print(policy)
    print(state_visit_count)
