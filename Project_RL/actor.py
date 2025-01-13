import typing
import numpy as np
import random

class Actor():
    def __init__ (self):
        pass
    def act(self):
        pass

class UniformBuyActor(Actor):
    def act(self, state) -> float:
        return 0.5

class TabularQActor(Actor):
    def __init__(self, environment_train, alpha=0.05, gamma=0.9, starting_epsilon=1, epsilon_decay_rate=0.999,
                min_epsilon=0.1, num_episodes=5000):
        # Parameters
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = starting_epsilon
        self.epsilon_decay_rate = epsilon_decay_rate
        self.min_epsilon = min_epsilon
        self.num_episodes = num_episodes
        #self.num_bins_price = 21  # Discretization bins for price
        self.bins = [0, 20, 40, 60, 80, 100, float('inf')]
        self.num_hours = 24
        self.actions = [-1, -0.5, 0, 0.5, 1]  # Discrete actions
        self.max_steps = len(environment_train.timestamps)
        self.storage_bins = [0, 30, 60, 90, 120]
        self.Q = np.zeros((len(self.bins), self.num_hours, len(self.storage_bins), len(self.actions)))
        self.environment_train = environment_train

    def train(self):
        print('Training the tabular Q-learning agent...')
        for episode in range(self.num_episodes):
            state = self.environment_train.reset()
            for t in range(self.max_steps):
                # Choose action using epsilon-greedy policy
                storage, price, hour, day = state
                price_bin_index = np.digitize(price, self.bins) - 1
                hour_index = int(hour-1)
                storage_index = np.digitize(storage, self.storage_bins) - 1
                if np.random.rand() < self.epsilon:
                    action_idx = random.randrange(len(self.actions)) # Explore
                else:
                    action_idx = np.argmax(self.Q[price_bin_index, hour_index, storage_index, :]) # Exploit

                # Take action, observe reward and next state
                action = self.actions[action_idx]
                next_state, reward, terminated = self.environment_train.step(action)
                next_storage, next_price, next_hour, _ = next_state
                next_price_bin_index = np.digitize(next_price, self.bins) - 1
                next_hour_index = int(next_hour-1)
                next_storage_index = np.digitize(next_storage, self.storage_bins) - 1
                # Update Q-value
                if (hour == 16 and storage == 30) or (14 > hour > 11 and storage < 80):
                    self.Q[price_bin_index, hour_index, storage_index, action_idx] = -10000000
                else:
                    best_next_action = np.argmax(self.Q[next_price_bin_index, next_hour_index, next_storage_index, :])
                    self.Q[price_bin_index, hour_index, storage_index, action_idx] += self.alpha * (reward + self.gamma * self.Q[next_price_bin_index, next_hour_index, next_storage_index, best_next_action] - self.Q[price_bin_index, hour_index, storage_index, action_idx])

                # Transition to next state
                state = next_state

            # Update epsilon value
            self.epsilon = self.epsilon * self.epsilon_decay_rate
            if self.epsilon < self.min_epsilon:
                self.epsilon = self.min_epsilon
            
            if episode % 100 == 0:
                print(f'-- Finished training {episode} episodes, epsilon = {self.epsilon}...')
        
    def act(self, state):
        storage, price, hour, _ = state
        price_bin_index = np.digitize(price, self.bins) - 1
        hour_index = int(hour-1)
        storage_index = np.digitize(storage, self.storage_bins) - 1

        # Select the best action (greedy policy)
        best_action_index = np.argmax(self.Q[price_bin_index, hour_index, storage_index, :])
        best_action = self.actions[best_action_index]

        return best_action
        