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

class HourlyMeansActor(Actor):
    def __init__(self):
        super().__init__()
        self.means = {'Buy':
                      [5, 4, 6, 3, 2, 7, 1, 24, 22, 23,
                       8, 17, 16, 9, 21, 15, 18, 14],
                      'Sell':
                      [20, 10, 13, 11, 19, 12]}

    def act(self, state):
        storage_level, price, hour, day = state
        if storage_level >= 130 and hour in self.means["Sell"]:
            return -1
        elif hour in self.means["Buy"]:
            return 1
        else:
            return 0

class AveragedBuyingActor(Actor):
    def __init__(self, amount_of_prices: int = 12, threshold: float = 0.1):
        self.current_average : float = 0
        self.price_queue = []
        #Amount of prices to be considered
        #24 prices to a day
        self.amount_of_prices = amount_of_prices
        #Deviation from average price to trigger buy sell action
        self.threshold = threshold

    def act(self, state):
        return_value = self.makeDecision(state[1])
        self.updateAverage(state[1])
        if state[0] > 160 and return_value == 1:
            return 0
        return return_value
    def updateAverage(self, newPrice:int):
        if len(self.price_queue) == self.amount_of_prices:
            self.current_average -= self.price_queue[0]/self.amount_of_prices
            self.current_average += newPrice/self.amount_of_prices
            self.price_queue.pop(0)
            self.price_queue.append(newPrice)
        else:
            self.price_queue.append(newPrice)
            self.current_average = sum(self.price_queue)/len(self.price_queue)
    def makeDecision(self, current_price:int) -> float:
        if self.current_average == 0:
            return 1
        difference = current_price - self.current_average
        fraction  = difference/self.current_average
        #TODO maybe make sure that
        if fraction > self.threshold:
            return -1
        if fraction * -1 > self.threshold:
            return 1

        return 0
        

class TabularQActor(Actor):
    def __init__(self, environment_train, environment_test, alpha=0.005, gamma=0.9, starting_epsilon=1, epsilon_decay_rate=0.9995,
                min_epsilon=0.1, num_episodes=5000):
        # Parameters
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = starting_epsilon
        self.epsilon_decay_rate = epsilon_decay_rate
        self.min_epsilon = min_epsilon
        self.num_episodes = num_episodes
        #self.num_bins_price = 21  # Discretization bins for price
        self.bins = [0, 20, 40, 60, 80, 100]
        self.num_hours = 24
        self.actions = [-1, -0.5, 0, 0.5, 1]  # Discrete actions
        self.max_steps = len(environment_train.timestamps)
        self.storage_bins = [0, 30, 60, 90, 120]
        self.Q = np.zeros((len(self.bins), self.num_hours, len(self.storage_bins), len(self.actions)))
        self.environment_train = environment_train
        self.environment_test = environment_test

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

                if action > 0:  # Buying action
                    reward += (24 - hour) * 0.1  # Encourage early purchases

                # Update Q-value
                if (14 > hour > 11 and storage < 80):
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
                val_reward = self.val()
                print(f'-- Finished training {episode} episodes, epsilon = {self.epsilon}, validation reward = {val_reward}...')
        
    def act(self, state):
        storage, price, hour, _ = state
        price_bin_index = np.digitize(price, self.bins) - 1
        hour_index = int(hour-1)
        storage_index = np.digitize(storage, self.storage_bins) - 1

        # Select the best action (greedy policy)
        best_action_index = np.argmax(self.Q[price_bin_index, hour_index, storage_index, :])
        best_action = self.actions[best_action_index]

        return best_action

    def val(self):
        aggregate_reward = 0
        terminated = False
        state = self.environment_test.reset()

        while not terminated:
            # agent is your own imported agent class
            action = self.act(state)
            #action = np.random.uniform(-1, 1)
            # next_state is given as: [storage_level, price, hour, day]
            next_state, reward, terminated = self.environment_test.step(action)
            state = next_state
            aggregate_reward += reward

        return aggregate_reward