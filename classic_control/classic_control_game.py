import gym
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(2)  # For repeatable experiments

Q = {}
gamma = 1


def discrete(state, weights=(1, 2)):
    """ Discretize the states """
    return tuple(np.round(attribute, weights[index]) for index, attribute in enumerate(state))


def argmax_action(state):
    """ For an input state - find the best action | action that has highest Q(S, A) """
    if state not in Q.keys():
        Q[state] = np.ones(3)
    return np.argmax(Q[state])


class GameAgent:
    def __init__(self, game_name='MountainCar-v0', round_states=(1, 2), iterations=10000, alpha=0.05, epsilon=0.04):
        """
        :param game_name: name of the game that you want to play
        :param round_states: rounds the variables of the state to this number of digits
        if round_states is None : a balanced weight
        eg. round_states = [1, 1]
        if states has two variables that is being observed
        :param iterations: how many episodes of the game to play
        """
        # self.game_name = "CartPole-v0"
        self.game_name = game_name
        self.env = gym.make(self.game_name)
        self.num_actions = self.env.action_space.n
        self.num_state_variables = self.env.observation_space.high.__len__()
        self.action_space = list(range(self.num_actions))
        self.total_episodes = iterations
        self.episode_rewards = []
        self.total_success = 0  # Number of episodes which it has successfully played
        self.alpha = alpha
        self.epsilon = epsilon  # Percentage of random exploration

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
            if np.random.uniform(0, 1) < self.epsilon:
                action = self.env.action_space.sample()
            else:
                # Choose a greedy action according to argmax of Q(state, action)
                action = argmax_action(state)

            new_state, reward, episode_over, _ = self.env.step(action)
            new_state = discrete(new_state, self.weights)
            episode_reward += reward

            if new_state not in Q.keys():
                Q[new_state] = np.ones(self.num_actions)

            # Q-learning formual
            Q[state][action] += self.alpha * (reward + gamma * Q[new_state][argmax_action(new_state)]
                                              - Q[state][action])

            state = new_state

        self.episode_rewards.append(episode_reward)

    def run(self):
        """ Play specified number of iterations of the game """
        for episode_count in range(self.total_episodes):
            self.play(episode_count)

            if (episode_count + 1) % 5000 == 0:
                print("Step : {} States : {}".format(episode_count, Q.items().__len__()))

                self.visualize_returns()

                # Evaluate the game every 100 steps
                if episode_count % 100 == 0:
                    self.evaluate(episode_count)

            # Decay epsilon
            if episode_count == 10000:
                self.epsilon = 0.01

    def evaluate(self, episode_count):
        """
        Evaluate the performance of the agent over the last 100 episodes.
        Helpful in seeing if the agent is converging or not
        """
        return_last_100 = self.episode_rewards[-100:]
        average_return_last100 = np.mean(return_last_100)
        print("Episode : {} \tAverage of last 100 episodes : {}".format(episode_count, average_return_last100))

    def visualize_returns(self):
        average_returns = np.asarray(self.episode_rewards).reshape(-1, 100)
        average_returns = np.mean(average_returns, axis=1)

        # Do a polynomail fit to find the underlying increment in the returns
        X = range(len(average_returns))
        coeffs = np.polyfit(X, average_returns, 3)
        Y = np.dot(np.vander(X, len(coeffs)), coeffs)

        plt.plot(X, average_returns)
        plt.plot(X, Y)
        plt.title("Average return for every 100 episode vs with alpha = {}".format(self.alpha))
        plt.show()


if __name__ == '__main__':
    # Grid search for parameter alpha
    for alpha in [0.1, 0.05, 0.01]:
        ga = GameAgent(iterations=15000, alpha=alpha)
        ga.run()
