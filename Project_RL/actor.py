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

class ExponentialMovingAverage(Actor):
    def __init__(self, alpha=0.1, number_of_sales=0):
        """
        Initialize the actor with exponential running averages for each hour.

        :param alpha: Smoothing factor for the exponential running average (0 < alpha <= 1).
        :param number_of_sales: Initial number of sales made (default 0, max 6).
        """
        super().__init__()
        self.alpha = alpha
        self.number_of_sales = number_of_sales
        self.current_state = None

        # Initialize a list to store running averages for each hour (24 hours) with a high default value
        self.means = [10**6] * 24

        # Initialize a list to store the last 29 purchase decisions
        self.last_purchases = []

    def get_plotting_data(self):
        assert self.current_state is not None, (
        "Remember to call .act() at least once before calling .get_plotting_data()")
        storage_levels, prices, hours, days = self.current_state
        mean = self.means[int(hours) - 1]
        return (storage_levels, prices, hours, days, mean)

    def _update_running_average(self, hour, price):
        """
        Update the exponential running average for a given hour and price.

        :param hour: Hour of the day (integer).
        :param price: Observed price at the given hour (float).
        """
        current_avg = self.means[hour]
        if current_avg == 10**6:
            # Initialize the running average if it's still at the default value
            self.means[hour] = price
        else:
            # Update the running average using the exponential formula
            self.means[hour] = self.alpha * price + (1 - self.alpha) * current_avg

    def _update_purchase_history(self, price):
        """
        Update the purchase history to maintain the last 29 purchase decisions.

        :param price: Price of the current purchase decision.
        """
        self.last_purchases.append(price)
        if len(self.last_purchases) > 29:
            self.last_purchases.pop(0)

    def act(self, state):
        """
        Decide the action based on storage level, price, and hour.

        :param state: Tuple (storage_level, price, hour, day).
        :return: Action to take: -1 (sell), 1 (buy), or 0 (hold).
        """
        self.current_state = state
        storage_level, price, hour, day = state
        hour = int(hour) - 1

        # Update the running average for the current hour
        self._update_running_average(hour, price)

        # Order the running averages (cheapest first)
        price_low_to_high_indexes = sorted(range(len(self.means)), key=lambda i: self.means[i])

        # Decision logic for buying
        if storage_level <= 160 and \
        hour in price_low_to_high_indexes[:12 + self.number_of_sales]:
            self._update_purchase_history(price)  # Track purchase decisions
            return 1  # Buy

        # Decision logic for selling
        elif self.number_of_sales != 0 and \
        hour in price_low_to_high_indexes[-self.number_of_sales:] and \
        price > max(self.last_purchases, default=10**6):
            return -1  # Sell

        else:
            return 0  # Hold


class SimpleMovingAverageActor(Actor):
    def __init__(self, amount_of_prices: int = 189, threshold: float = 0.1):
        self.current_average : float = 0
        self.price_queue = []
        #Amount of prices to be considered
        #24 prices to a day
        self.amount_of_prices = amount_of_prices
        #Deviation from average price to trigger buy sell action
        self.threshold = threshold
        self.current_state = None

    def get_plotting_data(self):
        assert self.current_state is not None, (
        "Remember to call .act() at least once before calling .get_plotting_data()")
        storage_levels, prices, hours, days = self.current_state
        upper_threshold = self.current_average + (self.current_average * self.threshold)
        lower_threshold = self.current_average - (self.current_average * self.threshold)
        return (storage_levels, prices, hours, days, self.current_average, upper_threshold, lower_threshold)

    def act(self, state):
        self.current_state = state
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
        return 0 # The actor needs to disproportionally buy more than sell, maybe this should be 1?
        

class TabularQActor(Actor):
    def __init__(self, environment_train, environment_test, amount_of_prices: int=161, threshold: float=0.1, alpha=0.005, gamma=0.9, starting_epsilon=1, epsilon_decay_rate=0.9995,
                min_epsilon=0.1, num_episodes=5000):
        # Parameters
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = starting_epsilon
        self.epsilon_decay_rate = epsilon_decay_rate
        self.min_epsilon = min_epsilon
        self.num_episodes = num_episodes
        #self.num_bins_price = 21  # Discretization bins for price
        #self.bins = [0, 20, 40, 60, 80, 100]
        self.bins = [-1.00001, -0.75, -0.5, -0.25, 0, 0.5, 1, 1.5] # bins for price difference from average
        self.num_hours = 24
        self.actions = [-1, -0.5, 0, 0.5, 1]  # Discrete actions
        self.max_steps = len(environment_train.timestamps)
        self.storage_bins = [0, 30, 60, 90, 120]
        self.Q = np.zeros((len(self.bins), self.num_hours, len(self.storage_bins), len(self.actions)))
        self.environment_train = environment_train
        self.environment_test = environment_test

        self.current_average : float = 0
        self.price_queue = []
        #Amount of prices to be considered: 161 is optimal window
        #24 prices to a day
        self.amount_of_prices = amount_of_prices
        #Deviation from average price to trigger buy sell action
        self.threshold = threshold

    def updateAverage(self, newPrice:int):
        if len(self.price_queue) == self.amount_of_prices:
            self.current_average -= self.price_queue[0]/self.amount_of_prices
            self.current_average += newPrice/self.amount_of_prices
            self.price_queue.pop(0)
            self.price_queue.append(newPrice)
        else:
            self.price_queue.append(newPrice)
            self.current_average = sum(self.price_queue)/len(self.price_queue)

    def train(self):
        print('Training the tabular Q-learning agent...')
        for episode in range(self.num_episodes):
            state = self.environment_train.reset()
            for t in range(self.max_steps):
                # Choose action using epsilon-greedy policy
                storage, price, hour, day = state
                #price_bin_index = np.digitize(price, self.bins) - 1
                # instead of price bins, use difference from moving average
                fraction = self.calculate_fraction(price)
                #print(fraction)
                self.updateAverage(price)

                price_bin_index = np.digitize(fraction, self.bins) - 1
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

                reward = self.calculate_reward(action,self.bins[price_bin_index],self.storage_bins[storage_index])
                #print(reward,action)
                # Update Q-value
                # if (14 > hour > 11 and storage < 80):
                #     pass
                #     #self.Q[price_bin_index, hour_index, storage_index, action_idx] = -10000000
                # else:
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
    def calculate_fraction(self,price):
        if self.current_average == 0:
            fraction = 0
        else:     
            difference = price - self.current_average
            fraction  = difference/self.current_average
        return fraction
    def calculate_reward(self,action,price_difference,storage_level,reward_parameter = 2.2):

        #print(f'current price = {price_difference}')
        #print(f'positive reward = {1* ((price_difference * -1) + (reward_parameter * ((120 - storage_level)/120)))}')
        #print(f'negative reward = {price_difference}')
        if action <= 0:
            return action * price_difference * -1
        else:
            return action * ((price_difference * -1) + (reward_parameter * ((120 - storage_level)/120)))

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
            storage, price, hour, day = state
            fraction = self.calculate_fraction(price)
            price_bin_index = np.digitize(fraction, self.bins) - 1
            storage_index = np.digitize(storage, self.storage_bins) - 1

            aggregate_reward += self.calculate_reward(action,self.bins[price_bin_index],self.storage_bins[storage_index])

        return aggregate_reward