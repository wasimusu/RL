import gym
import numpy as np

np.random.seed(2)

policy = dict()
Q = {}
alpha = 0.05
gamma = 1
epsilon = 0.04  # Percentage of random exploration
total_climbs = 0

# game = "CartPole-v0"
# action_space = [0, 1]  # Change this line to make things more general

game = 'MountainCar-v0'
action_space = [0, 1, 2]  # Change this line to make things more general

env = gym.make(game)


def discrete(state, weights=(1, 2)):
    """ Discretize the states """
    return tuple(np.round(attribute, weights[index]) for index, attribute in enumerate(state))


def argmax_action(state):
    """ For an input state - find the best action | action that has highest Q(S, A) """
    if state not in Q.keys():
        Q[state] = np.ones(3)
    return np.argmax(Q[state])


def updateQ(state, action, new_state, reward):
    # If the new_state, action does not exist, register it

    if new_state not in Q.keys():
        Q[new_state] = np.ones(3)

    Q[state][action] += alpha * (reward + gamma * Q[new_state][argmax_action(new_state)] - Q[state][action])


def play(step):
    episode_reward = 0
    global total_climbs

    # Start the game
    state = env.reset()
    state = discrete(state)

    if state not in Q.keys():
        Q[state] = np.ones(3)

    episode_over = False
    while not episode_over:
        # Do epsilon greedy exploitation
        if np.random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()
        else:
            # Choose a greedy action according to argmax of Q(state, action)
            action = argmax_action(state)

        new_state, reward, episode_over, _ = env.step(action)
        new_state = discrete(new_state)
        episode_reward += reward

        if new_state not in Q.keys():
            Q[new_state] = np.ones(3)

        updateQ(state=state, action=action, new_state=new_state, reward=reward)
        state = new_state

        if step >= 9000:
            env.render()

    if step >= 9000:
        env.close()  # Need to close if you render

    if episode_reward > -200:
        total_climbs += 1


if __name__ == '__main__':
    for step in range(10000):
        play(step)
        if step % 1000 == 0:
            print("Step : {} States : {} Total Climbs : ".format(step, Q.items().__len__()), total_climbs)
