import gym
import numpy as np

np.random.seed(2)

policy = dict()
Q = {}
alpha = 0.3
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
    """ For an input state - find the best action | action that has highest Q(S, A) """

    max_Q = -1
    optimal_action = env.action_space.sample()
    for action in action_space:
        temp_Q = Q.get((state, action), -1)

        # Q(S, A) may not exist (even for an existing state)
        if temp_Q == -1:
            Q[(state, action)] = 0

        if temp_Q > max_Q:
            optimal_action = action
            max_Q = temp_Q

        # print(state, action, max_Q)

    # print(state, optimal_action, Q[(state, optimal_action)])
    return optimal_action


def updateQ(state, action, new_state, reward):
    # If the new_state, action does not exist, register it
    if Q.get((new_state, action), -1) == -1:
        Q[(new_state, action)] = np.random.uniform(0, 1)

    Q[(state, action)] += alpha * (reward + discount * Q[(new_state, argmax_action(new_state))] - Q[(state, action)])


def play(step):
    episode_reward = 0

    # Start the game
    state = env.reset()
    state = discrete(state)

    action = env.action_space.sample()
    if Q.get((state, action), -1) == -1:
        Q[(state, action)] = np.random.uniform(0, 1)

    episode_over = False
    while not episode_over:
        # Do epsilon greedy exploitation
        if np.random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()
        else:
            # Choose a greedy action according to argmax of Q(state, action)
            # print("Greedy copy ignore")
            action = argmax_action(state)

        # print(state, action)

        new_state, reward, episode_over, _ = env.step(action)
        new_state = discrete(new_state)
        episode_reward += reward

        updateQ(state=state, action=action, new_state=new_state, reward=reward)
        # print("Updated : ", Q[(state, action)])
        state = new_state

        if step >= 49000:
            env.render()

    if step >= 49000:
        env.close()  # Need to close if you render
    if episode_reward != -200:
        print("Episode {} Reward : {}".format(step, episode_reward))


if __name__ == '__main__':
    for step in range(50000):
        play(step)
        if step % 1000 == 0:
            print("Step : {} States : {}".format(step, Q.items().__len__()))
