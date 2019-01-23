""" Implements policy iteration on Gamblers' Problem """

import numpy as np
import matplotlib.pyplot as plt

num_states = 100
value = np.zeros(num_states)
value[num_states - 1] = 1.0

policy = np.zeros(num_states)  # means that you bet 0 at each state

# Define coin
pHead = 0.4
pTail = 1 - pHead


def argmax_action(state):
    """ Find the best action for a particular state """

    # Try out all the (possible) action and choose the best action
    max_q, best_action = -1, -1
    # If we bet 0 we're not going to change anything
    for action in range(1, min(state, num_states - state - 1) + 1):
        q = pHead * value[state + action] + pTail * value[state - action]
        if q >= max_q:
            best_action = action
            max_q = q
    return best_action


def policy_evaluation():
    while True:
        delta = 0
        # Use policy to compute new value for each state
        for state in range(0, num_states - 1):  # Do not update the fine value-state
            previous_value = value[state]
            action = int(policy[state])  # Take action according to policy for a given state and evaluate its' value
            value[state] = pHead * value[state + action] + pTail * value[state - action]
            delta = max(delta, abs(value[state] - previous_value))

        if delta < 1e-20:
            break


def policy_iteration():
    # If we start at state 0, it quits because that's at the best state
    for state in range(1, num_states):
        previous_action = policy[state]
        best_action = argmax_action(state)
        if previous_action == best_action:
            break
        else:
            policy[state] = best_action
            policy_evaluation()

    plt.plot(range(num_states), policy, drawstyle="steps")
    plt.show()


if __name__ == '__main__':
    policy_iteration()
