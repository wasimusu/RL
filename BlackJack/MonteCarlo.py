import numpy as np
import matplotlib.pyplot as plt

HIT, STICK = 1, 0
actions = [HIT, STICK]
USEABLE_ACE = [1, 0]
deck = [2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10, 11]

# 12-21 * USEABLE_ACE * DEALER 2-11 : 10 * 2 * 10
states = []
states_action = []
for player_sum in range(12, 21 + 1):
    for ace in USEABLE_ACE:
        for dealer_card in range(2, 11 + 1):
            state = "P_{}_A_{}_D_{}".format(player_sum, ace, dealer_card)
            states.append(state)
            for action in actions:
                states_action.append("{}_S_{}".format(state, action))

Q = dict(zip(states_action, np.random.uniform(0, 1, 400)))  # 400 = Number of States * Number of Action
policy = dict(zip(states, np.random.randint(0, 1, 200)))  # 200 = One action that can be taken from each state


def update_Q(state, reward):
    """ Apply incremental update to Q values for given state """
    pass


def argmax_action(state):
    """ Find the best action which leads to the highest Q for a given state """
    return 1


def generate_game(num_cards=1):
    deck = [2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10, 11]
    dealer = [np.random.choice(deck, num_cards)]
    player = [np.random.choice(deck, num_cards)]
    return dealer, player


def draw_card(num_cards=1):
    return np.random.choice(deck, num_cards)


def sum_deck(deck):
    sum_deck = sum(deck)
    if 11 in deck:
        if sum_deck > 21:
            sum_deck -= 10
    return sum_deck


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


def evaluate_game(dealer, player):
    """ Returns reward for by comparing dealer's and players' hand """

    # Bust case
    sum_dealer = sum_deck(dealer)
    sum_player = sum_deck(player)

    # Both of them scoring equal or both of them being bust
    if sum_dealer == sum_player:
        return 0

    # Dealer bust
    if sum_dealer > 21:
        return 1

    # Player bust
    if sum_player > 21:
        return -1

    # Normal outcomes
    if sum_dealer > sum_player:
        return -1
    else:
        return 1


def generate_states(cards, useable_ace, dealer_first_card):
    pass


def generate_episode():
    # Draw cards until the player's hand sum to 12
    dealer_hand, player_hand = [], []
    while True:
        dealer_hand.append(draw_card())
        player_hand.append(draw_card())
        if sum_deck(player_hand) > 11: break

    # Dealer's first card is visible
    dealer_first_card = dealer_hand[0]
    while True:
        useable_ace = 1 if 11 in player_hand else 0
        player_state = "P_{}_A_{}_D_{}".format(sum_deck(player_hand), useable_ace, dealer_first_card)
        # Play according to policy
        if policy[player_state] == HIT:  # Draw card
            dealer_hand.append(draw_card())
            player_hand.append(draw_card())
        else:
            # Evaluate the game for reward
            reward = evaluate_game(dealer_hand, player_hand)
            episode_states = generate_states(player_hand, useable_ace, dealer_first_card)

            for ep_state in episode_states:
                update_Q(ep_state, reward)
                # Update

    # Update the policy after each episode
    # Do argmax action for each state
    for state in states:
        policy[state] = argmax_action(state)
