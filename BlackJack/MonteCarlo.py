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
Q["P_>22"] = 0  # policy for state greater than 21

policy = dict(zip(states, np.random.randint(0, 1, 200)))  # 200 = One action that can be taken from each state
policy["P_>22"] = 0  # policy for state greater than 21
num_state_visits = dict(zip(states, [0] * len(states)))  # Keeping count of the visit for each state


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


def update_Q(state, action, reward):
    """ Apply incremental update to Q values for given state """
    num_state_visits[state] += 1
    state_action = "{}_S_{}".format(state, action)
    Q[state_action] = Q[state_action] + (1.0 / num_state_visits[state])


def argmax_action(state, reward=1):
    """ Find the best action which leads to the highest Q for a given state.

    # From ech state I can only take two actions : HIT / STICK
    # I have Q values of both the state
    # Should I just average them to find argmax ?
     """
    max_q, optimal_action = -1, -1
    # Find the action leading to the highest Q
    for action in [0, 1]:
        state_action = "{}_S_{}".format(state, action)

        if action == 1:
            drawn_card = draw_card()
            a, old_sum, *c = state_action.split("_")
            new_sum = sum_deck([int(old_sum), drawn_card])
            if new_sum > 21:
                state_action = "P_>22"
            else:
                state_action = a + "_" + new_sum.__str__() + "_" + "_".join(c)

        if Q[state_action] > max_q:
            optimal_action = action
            max_q = Q[state_action]

    return optimal_action


def generate_states_for_cards(cards, useable_ace, dealer_first_card):
    """
    >> generate_states_for_cards([2,2, 3, 5], 1, 10)
    >> ['P_2_A_1_D_10', 'P_4_A_1_D_10', 'P_7_A_1_D_10', 'P_12_A_1_D_10']
    """
    return ["P_{}_A_{}_D_{}".format(sum_deck(cards[:i]), useable_ace, int(dealer_first_card)) for i in
            range(2, len(cards) + 1)]


aa = generate_states_for_cards([2, 2, 3, 5], 1, 10)
print(aa)

def generate_episode():
    # Draw cards until the player's hand sum to 12

    dealer_hand, player_hand = [], []
    while True:
        dealer_hand.append(draw_card())
        player_hand.append(draw_card())
        if sum_deck(player_hand) > 11: break

    # If the sum if 21, the player won

    # Dealer's first card is visible
    dealer_first_card = dealer_hand[0]

    while True:
        useable_ace = 1 if 11 in player_hand else 0
        player_state = "P_{}_A_{}_D_{}".format(int(sum_deck(player_hand)), useable_ace, int(dealer_first_card))

        print("State : ", player_state)
        # Play according to policy
        if policy[player_state] == HIT:  # Draw card
            dealer_hand.append(draw_card())
            player_hand.append(draw_card())
        else:
            # Evaluate the game for reward
            reward = evaluate_game(dealer_hand, player_hand)
            episode_states = generate_states_for_cards(player_hand, useable_ace, dealer_first_card)

            print("\n".join(episode_states))
            for ep_state in episode_states:
                update_Q(ep_state, 1, reward)
            break

    # Update the policy after each episode
    # Do argmax action for each state
    for state in states:
        policy[state] = argmax_action(state)


print(generate_episode())
