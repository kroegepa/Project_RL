import typing
import numpy as np
import random
import matplotlib.pyplot as plt
from collections import deque
from matplotlib.cm import get_cmap
from mpl_toolkits.mplot3d import Axes3D
import inflect
o_suffix = inflect.engine()
import csv



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
        self.sell_times = 0
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = starting_epsilon
        self.epsilon_decay_rate = epsilon_decay_rate
        self.min_epsilon = min_epsilon
        self.num_episodes = num_episodes
        #self.num_bins_price = 21  # Discretization bins for price
        #self.bins = [0, 20, 40, 60, 80, 100]
        self.bins = [-1.00001, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 1] # bins for price difference from average
        self.num_hours = 24
        self.actions = [-1, -0.5, 0, 0.5, 1]  # Discrete actions
        self.max_steps = len(environment_train.timestamps) * 24 - 1
        self.storage_bins = [0, 30, 60, 90, 120]
        self.Q = np.zeros((len(self.bins), self.num_hours, len(self.storage_bins), len(self.actions)))
        self.environment_train = environment_train
        self.environment_test = environment_test

        self.current_moving_average : float = 0
        self.current_moving_average_val : float = 0
        self.price_queue = []
        self.price_queue_val = []
        #Amount of prices to be considered: 161 is optimal window
        #24 prices to a day
        self.amount_of_prices = amount_of_prices
        #Deviation from average price to trigger buy sell action
        self.threshold = threshold

    def visualize_trajectory(self, states, from_indx_, to_indx_, nmbr_days_in_fig=3, animate=False, spam_protection=True):
        end = to_indx_ if to_indx_ is None else to_indx_*24
        window = states[from_indx_*24:end]
        if spam_protection:
            if animate:
                window = window[-2*24:]
            else:
                window = window[-6*24:]
        
        if animate:
            print("This code broke when the reward plot was added and will not be fixed unless necessary.")
            """self.fig, self.ax1 = plt.subplots(figsize=(12, 6))  # Standard height
            self.ax2 = self.ax1.twinx()
            self.time_labels = [h + 1 for h in range(24)]
            self.plot_prices = deque([0] * 24, maxlen=24)  # Prices for the last 24 hours
            self.plot_storages = deque([0] * 24, maxlen=24)  # Storage levels for the last 24 hours
            self.plot_actions = deque([0] * 24, maxlen=24)  # Actions for the last 24 hours
            self.plot_hours = deque(range(24), maxlen=24)  # Always keep 0-23 for the hours
            self.running_average_plot = deque([0] * 24, maxlen=24)  # Running average data
            self.reward_plot = deque([0] * 24, maxlen=24)
            for state in window:
                price, hour, day, storage, action, running_average, reward_plot = state
                hour = int(hour)

                # Update the last 24 hours of data
                self.plot_prices[hour - 1] = price
                self.plot_storages[hour - 1] = storage
                self.plot_actions[hour - 1] = action
                self.running_average_plot[hour - 1] = running_average
                self.reward_plot[hour - 1] = reward_plot

                # Clear axes for redraw
                self.ax1.cla()
                message = f": {int(day)}"
                self.trajectory_plotter(message, (0.5, 0.02), 0.22)
                plt.pause(0.2)"""

        else:
            for first_state in range(0, len(window), nmbr_days_in_fig*24):
                # Update the last 24 hours of data
                self.fig, self.ax1 = plt.subplots(figsize=(nmbr_days_in_fig*4, 8))
                self.ax2 = self.ax1.twinx()
                self.plot_prices, hours, days, self.plot_storages, self.plot_actions, self.running_average_plot, self.reward_plot = list(zip(*window[first_state:first_state+nmbr_days_in_fig*24]))

                self.time_labels = []
                for hour, day in zip(hours, days):
                    if int(hour) == 24:
                        self.time_labels.append(f"{o_suffix.ordinal(int(day)+1)} day")
                    else:
                        self.time_labels.append(str(int(hour)))

                message = f"s: {int(days[0])} to {int(days[-1])}"
                self.trajectory_plotter(message, (0.5, 0.018), 0.2)
            plt.show()

    def trajectory_plotter(self, message, bbox_to_anchor, bottom_space):
        """
        Visualize data dynamically by displaying graphs frame by frame.

        Parameters:
            price (float): Price at the given hour.
            hour (int): Current hour.
            day (int): Day number.
            storage (float): Storage level at the given hour.
            action (float): Action taken at the given hour.
            running_average (float): Running average of prices.
            Q_vals (tensor): Placeholder for additional graph data.
        """
        # Bar chart for storage levels
        self.ax1.bar(range(len(self.plot_storages)), self.plot_storages, color="orange", alpha=0.6, label="Storage Level")

        # Arrows for actions
        action_colors = {
            -1: 'darkred',
            -0.5: 'magenta',
            0: 'purple',
            0.5: 'teal',
            1: 'darkblue'
        }
        action_labels = {
            -1: "Sell: -1",
            -0.5: "Sell: -0.5",
            0: "No Action",
            0.5: "Buy: 0.5",
            1: "Buy: 1"
        }

        # Add dummy plots for action labels
        for a, color in action_colors.items():
            self.ax1.hlines(0, 0, 0, colors=color, linewidth=4, label=action_labels[a])

        for i, (s, a) in enumerate(zip(self.plot_storages, self.plot_actions)):
            if a == 0:
                self.ax1.hlines(s, i - 0.4, i + 0.4, colors=action_colors[a], linewidth=4)
            elif a in action_colors:
                self.ax1.arrow(i, s, 0, 10 * a, color=action_colors[a], head_width=0.2, head_length=2, length_includes_head=True, linewidth=2)

        # Plot prices
        self.ax1.plot(range(len(self.plot_prices)), self.plot_prices, 'o', markerfacecolor="white", markeredgecolor="black", markeredgewidth=2, label="Price")

        # Plot running average
        self.ax1.plot(range(len(self.running_average_plot)), self.running_average_plot, '--', color="black", label="Running Average")

        self.ax2.plot(range(len(self.reward_plot)), self.reward_plot, '-', color='green', label='Reward')

        # Title
        self.ax1.set_title(f"Tabular Q learning trajectory, day{message}")

        # Labels
        self.ax1.set_xlabel("Time (hours)")
        self.ax1.set_ylabel("Price & Storage Level")
        self.ax2.set_ylabel("Reward")
        self.ax1.set_xticks(range(len(self.time_labels)))
        self.ax1.set_xticklabels(self.time_labels, rotation=60, ha="right", fontsize=9)

        # Combine legends
        handles_ax1, labels_ax1 = self.ax1.get_legend_handles_labels()
        handles_ax2, labels_ax2 = self.ax2.get_legend_handles_labels()

        handles = handles_ax1 + handles_ax2
        labels = labels_ax1 + labels_ax2

        self.fig.legend(handles, labels, loc="lower center", bbox_to_anchor=bbox_to_anchor, ncol=len(labels), frameon=True)

        # Adjust layout and display the graph
        plt.tight_layout()
        plt.subplots_adjust(bottom=bottom_space)  # Add space at the bottom for the legend



    def visualize_q_values(self, color_mapping_dim: str):
        """
        Visualizes the Q-values tensor in a 3D plot.

        :param color_mapping_dim: The dimension to be represented by color. Must be one of 'storage_bins' or 'actions'.
        """
        # Dimension mapping
        dims = {
            'price_bins': 0,
            'hours': 1,
            'storage_bins': 2,
            'actions': 3
        }

        action_labels = {
            0: "Sell: -1", 
            1: "Sell: -0.5", 
            2: "No Action", 
            3: "Buy: 0.5", 
            4: "Buy: 1" 
        }

        storage_labels = { 
            0: "0 to 30", 
            1: "30 to 60", 
            2: "60 to 90", 
            3: "90 to 120", 
            4: "120 and up" 
        }

        if color_mapping_dim not in ['storage_bins', 'actions']:
            raise ValueError("color_mapping_dim must be one of 'storage_bins' or 'actions'")

        color_dim_index = dims[color_mapping_dim]
        other_dim = 'storage_bins' if color_mapping_dim == 'actions' else 'actions'
        other_dim_index = dims[other_dim]

        # Get the range of values for the color dimension
        color_bins = self.Q.shape[color_dim_index]
        other_bins = self.Q.shape[other_dim_index]

        fig = plt.figure(figsize=(18, 12))  # Increased figure size for more spacing

        # Set up a 2x3 grid for subplots (2 on the top row, 3 on the bottom row)
        rows, cols = 2, 3
        grid_positions = [(row, col) for row in range(rows) for col in range(cols)]

        for other_bin in range(other_bins):
            row, col = grid_positions[other_bin]
            ax = fig.add_subplot(rows, cols, row * cols + col + 1, projection='3d')

            # Store surfaces and their max/min values for sorting
            surfaces = []

            for color_bin in range(color_bins):
                # Extract the corresponding slice
                if color_mapping_dim == 'storage_bins':
                    data_slice = self.Q[:, :, color_bin, other_bin]
                else:
                    data_slice = self.Q[:, :, other_bin, color_bin]

                # Calculate the max value for sorting
                max_value = np.max(data_slice)
                min_value = np.min(data_slice)

                # Store the data for sorting
                surfaces.append((max_value, min_value, color_bin, data_slice))

            # Sort surfaces by max/min values
            surfaces.sort(key=lambda x: (x[0], x[1]))  # Sort by max, then min values

            for _, _, color_bin, data_slice in surfaces:
                # Create the mesh grid for price_bins and hours
                price_bins, hours = np.meshgrid(
                    np.arange(self.Q.shape[dims['price_bins']]),
                    np.arange(self.Q.shape[dims['hours']])
                )

                # Normalize color mapping values
                norm = plt.Normalize(vmin=0, vmax=color_bins - 1)
                cmap = get_cmap('coolwarm')
                color_value = norm(color_bin)

                # Create facecolors as a 2D array matching the shape of data_slice
                facecolors = np.empty(data_slice.T.shape + (4,), dtype=float)
                facecolors[..., :] = cmap(color_value)

                # Plot the surface with a zorder
                ax.plot_surface(
                    hours, price_bins, data_slice.T, facecolors=facecolors, shade=False, alpha=0.7, rstride=1, cstride=1
                )

            # Update the title to use descriptive labels
            if other_dim == 'actions':
                title_label = action_labels.get(other_bin, f'Action {other_bin}')
            else:
                title_label = storage_labels.get(other_bin, f'Storage {other_bin}')

            ax.set_title(f'{other_dim.capitalize()} Bin: {title_label}', pad=20)  # Added padding for title
            ax.set_xlabel('Hours')
            ax.set_ylabel('Price Bins')
            ax.set_zlabel('Q-Values')

        # Add a legend in the empty grid position (last subplot)
        legend_ax = fig.add_subplot(rows, cols, len(grid_positions))  # Last grid position
        legend_ax.axis('off')  # Turn off axis for the legend box

        # Create the color legend
        norm = plt.Normalize(vmin=0, vmax=color_bins - 1)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])

        # Map color_bins to labels
        labels = action_labels if color_mapping_dim == 'actions' else storage_labels
        ticks = np.arange(color_bins)
        cbar = fig.colorbar(sm, ax=legend_ax, orientation='vertical', fraction=0.025, pad=0.1, ticks=ticks)
        cbar.ax.set_yticklabels([labels.get(i, f"{color_mapping_dim.capitalize()} {i}") for i in ticks])
        cbar.set_label(f'{color_mapping_dim.capitalize()} Values', rotation=90, labelpad=10)

        plt.subplots_adjust(wspace=0.4, hspace=0.4)  # Added space between subplots
        plt.show()

    def plot_rewards_during_training(self, reward_list):
        internal_reward, external_reward = zip(*reward_list)
        self.fig, self.ax1 = plt.subplots(figsize=(12, 8))
        self.ax2 = self.ax1.twinx()
        self.ax1.plot(range(len(internal_reward)), internal_reward, "-", color="blue", label="Internal Reward")
        self.ax2.plot(range(len(external_reward)), external_reward, "-", color="red", label="External Reward")
        self.ax1.set_ylabel("Internal Reward")
        self.ax2.set_ylabel("External Reward")
        plt.title("Validation Reward per Episode of Training")
        plt.xlabel("Episodes")
        # Combine legends
        handles_ax1, labels_ax1 = self.ax1.get_legend_handles_labels()
        handles_ax2, labels_ax2 = self.ax2.get_legend_handles_labels()

        handles = handles_ax1 + handles_ax2
        labels = labels_ax1 + labels_ax2
        self.fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.02), ncol=len(labels), frameon=True)
        plt.show()

    def updateAverage(self, newPrice:int, train=True):
        if train == True:
            if len(self.price_queue) == self.amount_of_prices:
                self.current_moving_average -= self.price_queue[0]/self.amount_of_prices
                self.current_moving_average += newPrice/self.amount_of_prices
                self.price_queue.pop(0)
                self.price_queue.append(newPrice)
            else:
                self.price_queue.append(newPrice)
                self.current_moving_average = sum(self.price_queue)/len(self.price_queue)
        else:
            if len(self.price_queue_val) == self.amount_of_prices:
                self.current_moving_average_val -= self.price_queue_val[0]/self.amount_of_prices
                self.current_moving_average_val += newPrice/self.amount_of_prices
                self.price_queue_val.pop(0)
                self.price_queue_val.append(newPrice)
            else:
                self.price_queue_val.append(newPrice)
                self.current_moving_average_val = sum(self.price_queue_val)/len(self.price_queue_val)

    def train(self):
        print('Training the tabular Q-learning agent...')
        validation_rewards = []
        external_rewards = []
        for episode in range(self.num_episodes):
            #print(f"Episode {episode}")
            state = self.environment_train.reset()
            for t in range(self.max_steps):
                # Choose action using epsilon-greedy policy
                storage, price, hour, day = state
                #price_bin_index = np.digitize(price, self.bins) - 1
                # instead of price bins, use difference from moving average
                fraction = self.calculate_fraction(price)
                #print(fraction)
                self.updateAverage(price)
                #print(f'Current average: {self.current_average}')

                price_bin_index = np.digitize(fraction, self.bins) - 1
                #print(f"Fraction: {fraction}, price bin: {price_bin_index}")
                hour_index = int(hour-1)
                if hour_index == 0:
                    self.sell_times = 0
                storage_index = np.digitize(storage, self.storage_bins) - 1
                if np.random.rand() < self.epsilon:
                    #print("Explore")
                    action_idx = random.randrange(len(self.actions)) # Explore
                else:
                    #print("Exploit")
                    action_idx = np.argmax(self.Q[price_bin_index, hour_index, storage_index, :]) # Exploit
                # Take action, observe reward and next state
                action = self.actions[action_idx]
                if action < 0:
                    self.sell_times += 1
                #print(f'Action: {action}')


                next_state, reward, terminated = self.environment_train.step(action)
                next_storage, next_price, next_hour, _ = next_state
                next_fraction = self.calculate_fraction(next_price)
                next_price_bin_index = np.digitize(next_fraction, self.bins) - 1
                next_hour_index = int(next_hour-1)
                next_storage_index = np.digitize(next_storage, self.storage_bins) - 1

                reward = self.calculate_reward(action,self.bins[price_bin_index],self.storage_bins[storage_index])
                #print(reward,action)
                # Update Q-value
                # if (14 > hour > 11 and storage < 80):
                #     pass
                #     #self.Q[price_bin_index, hour_index, storage_index, action_idx] = -10000000
                # else:

                #print(f"Before Update: Q[{price_bin_index}, {hour_index}, {storage_index}, {action_idx}] = {self.Q[price_bin_index, hour_index, storage_index, action_idx]}")
                
                best_next_action = np.argmax(self.Q[next_price_bin_index, next_hour_index, next_storage_index, :])
                #print(f"Reward: {reward}, Best Next Q: {self.Q[next_price_bin_index, next_hour_index, next_storage_index, best_next_action]}")
                self.Q[price_bin_index, hour_index, storage_index, action_idx] += self.alpha * (reward + self.gamma * self.Q[next_price_bin_index, next_hour_index, next_storage_index, best_next_action] - self.Q[price_bin_index, hour_index, storage_index, action_idx])

                #print(f"After Update: Q[{price_bin_index}, {hour_index}, {storage_index}, {action_idx}] = {self.Q[price_bin_index, hour_index, storage_index, action_idx]}")
                #print(f'Epsilon: {self.epsilon}')

                # Transition to next state
                state = next_state

            # Update epsilon value
            self.epsilon = self.epsilon * self.epsilon_decay_rate
            if self.epsilon < self.min_epsilon:
                self.epsilon = self.min_epsilon
            
            val_r = self.val()
            validation_rewards.append(val_r[0])
            external_rewards.append(val_r[1])
            if episode % 10 == 0:
                print(f'-- Finished training {episode} episodes, epsilon = {round(self.epsilon, 4)}\n' \
                      f'internal validation reward = {round(validation_rewards[-1], 1):,}\n' \
                      f'external validation reward = {round(external_rewards[-1], 1):,}')
        with open("tab_q_val_rewards.csv", "w") as f:
            writer = csv.writer(f)
            writer.writerow(["Validation Rewards", "External Rewards"])
            for internal, external in zip(validation_rewards, external_rewards):
                writer.writerow([internal, external])
        self.visualize_q_values('storage_bins')
        self.visualize_q_values('actions')
        
    def act(self, state):
        storage, price, hour, _ = state
        fraction = self.calculate_fraction(price, train=False)
        self.updateAverage(price, train=False)
        price_bin_index = np.digitize(fraction, self.bins) - 1
        hour_index = int(hour-1)
        if hour_index == 0:
            self.sell_times = 0
        storage_index = np.digitize(storage, self.storage_bins) - 1
        #print(f"Price: {price}, fraction: {fraction}, current average: {self.current_average_val}, price bin: {price_bin_index}")

        # Select the best action (greedy policy)
        #print(f"Q[{price_bin_index}, {hour_index}, {storage_index}, actions] = {self.Q[price_bin_index, hour_index, storage_index, :]}")
        best_action_index = np.argmax(self.Q[price_bin_index, hour_index, storage_index, :])
        best_action = self.actions[best_action_index]
        if best_action < 0:
            self.sell_times += 1
        #print(f"Best action index: {best_action_index}, action: {best_action}")
        return best_action

    def calculate_fraction(self, price, train=True):
        if train == True:
            if self.current_moving_average == 0:
                fraction = 0
            else:     
                difference = price - self.current_moving_average
                fraction  = difference/self.current_moving_average
            return fraction
        else:
            if self.current_moving_average_val == 0:
                fraction = 0
            else:
                difference = price - self.current_moving_average_val
                fraction = difference/self.current_moving_average_val
            return fraction

    def calculate_reward(self,action,price_difference,storage_level,reward_parameter = 4.2):

        #print(f'current price = {price_difference}')
        #print(f'positive reward = {1* ((price_difference * -1) + (reward_parameter * ((120 - storage_level)/120)))}')
        #print(f'negative reward = {price_difference}')

            
        if action < 0: # sell
            if self.sell_times >= 4:
                return -10
            else:
                return action * (price_difference - 0.2)* -1
        
        elif action == 0: # do nothing
            return 0
        
        elif action > 0: # buy
            if storage_level > 160:
                return -100
            return action * ((price_difference * -1) + (reward_parameter * (max(0,(130 - storage_level)/130))))

    def val(self):
        aggregate_reward = 0
        aggregate_test_reward = 0
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
            aggregate_test_reward += reward

        return aggregate_reward, aggregate_test_reward