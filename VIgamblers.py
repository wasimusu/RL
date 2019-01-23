""" Author : Professor """

# value iteration
import matplotlib.pyplot as plt
import numpy as np

v = np.zeros(101)
v[100] = 1.0  # terminal reward
pHeads = 0.4

# def argmax_action(state):
#     """ Find the best action for a particular state """
#
#     # Try out all the (possible) action and choose the best action
#     max_q, best_action = -1, -1
#     # If we bet 0 we're not going to change anything
#     # for action in range(1, min(state, 100 - state - 1)):
#     for action in range(min(state, (100-state)), 0, -1): #all possible bets
#         q = pHeads * v[state + action] + (1-pHeads) * v[state - action]
#         if q >= max_q:
#             best_action = action
#             max_q = q
#     return best_action

def valueIteration():
    # initialize the v table
    delta = 0
    while True:
        delta = 0
        # loop through all possible states
        for s in range(1,100):
            vOld = v[s]
            best = -1 
            # Bellman equation, loop through all possible actions maximizing
            for a in range(1,1 + min(s, (100-s))): #all possible bets
                # for this action, two outcomes, heads or tails
                # if heads get the money so next state is s+a
                # if tails lose the money so next state is s-a
                best = max(best, pHeads*v[s+a] + (1-pHeads)*v[s-a])
            # update the value of s with the new value
            v[s] = best
            delta = max(delta, abs(vOld - best))
        if delta < 1e-10:
            break


    # compute policy
    pi = np.zeros(101)
    # for each state
    for s in range(1,100):
        best = -1 
        # try each action and do an argmax
        # note, to get the same graph as in the book, start with the largest
        # bets and use >= so policy will pick the smalest equivalent bet
        # for a in range(min(s, (100-s)), 0, -1): #all possible bets
        for a in range(1, min(s, (100 - s))+1):  # all possible bets
            # Note the round! So very similar propabilities look the same
            now = round(pHeads*v[s+a] + (1-pHeads)*v[s-a],6)
            if now >= best:
                best = now
                bestAction = a
        pi[s] = bestAction
    #plt.plot(v, drawstyle="steps")
    plt.plot(pi, drawstyle="steps")
    plt.show()
    
valueIteration()
    
            

