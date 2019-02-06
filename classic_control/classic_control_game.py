import gym
import numpy as np

np.random.seed(2)  # For repeatable experiments

Q = {}
alpha = 0.05
gamma = 1
epsilon = 0.04  # Percentage of random exploration


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


class GameAgent:
    def __init__(self, game_name='MountainCar-v0', round_states=None, iterations=10000):
        """
        :param game_name: name of the game that you want to play
        :param round_states: rounds the variables of the state to this number of digits
        if round_states is None : a balanced weight
        eg. round_states = [1, 1]
        if states has two variables that is being observed
        :param iterations: how many episodes of the game to play
        """
        self.game_name = game_name
        self.game_name = "CartPole-v0"
        self.env = gym.make(self.game_name)
        self.num_actions = self.env.action_space.n
        self.num_state_variables = self.env.observation_space.high.__len__()
        self.action_space = list(range(self.num_actions))
        self.total_episodes = iterations
        self.episode_rewards = []
        self.total_success = 0  # Number of episodes which it has successfully played

        if round_states == None:
            round_states = [2] * self.num_state_variables
        if len(round_states) != self.num_state_variables:
            raise ValueError('Length of weights sould match variables in the observation / state ')
        self.weights = round_states

    def play(self, episode_count):
        """
        :param episode_count: What is the current episode we are playing
        """
        episode_reward = 0

        # Start the game
        state = self.env.reset()
        state = discrete(state, self.weights)

        if state not in Q.keys():
            Q[state] = np.ones(self.num_actions)

        episode_over = False
        while not episode_over:
            # Do epsilon greedy exploitation
            if np.random.uniform(0, 1) < epsilon:
                action = self.env.action_space.sample()
            else:
                # Choose a greedy action according to argmax of Q(state, action)
                action = argmax_action(state)

            new_state, reward, episode_over, _ = self.env.step(action)
            new_state = discrete(new_state, self.weights)
            episode_reward += reward

            if new_state not in Q.keys():
                Q[new_state] = np.ones(self.num_actions)

            updateQ(state=state, action=action, new_state=new_state, reward=reward)
            state = new_state

            if episode_count >= 9000:
                self.env.render()

        if episode_count >= 9000:
            self.env.close()  # Need to close if you render

        # if episode_reward > 20:
        #     self.total_success += 1

        if episode_reward > -200:
            self.total_success += 1

        # print(episode_count, episode_reward)

    def run(self):
        """ Play specified number of iterations of the game """
        for episode_count in range(self.total_episodes):
            self.play(episode_count)
            if episode_count % 1000 == 0:
                print("Step : {} States : {} Total Success : ".format(episode_count, Q.items().__len__()),
                      self.total_success)


if __name__ == '__main__':
    ga = GameAgent(iterations=1000000)
    ga.run()
