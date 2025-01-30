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
from numpy.random import default_rng



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
        super().__init__()
        self.alpha = alpha
        self.number_of_sales = number_of_sales
        self.current_state = None
        self.means = [10**6] * 24
        self.last_purchases = []

    def get_plotting_data(self):
        assert self.current_state is not None, (
        "Remember to call .act() at least once before calling .get_plotting_data()")
        storage_levels, prices, hours, days = self.current_state
        mean = self.means[int(hours) - 1]
        return (storage_levels, prices, hours, days, mean)

    def _update_running_average(self, hour, price):
        current_avg = self.means[hour]
        if current_avg == 10**6:
            self.means[hour] = price
        else:
            self.means[hour] = self.alpha * price + (1 - self.alpha) * current_avg

    def _update_purchase_history(self, price):
        self.last_purchases.append(price)
        if len(self.last_purchases) > 29:
            self.last_purchases.pop(0)

    def act(self, state):
        self.current_state = state
        storage_level, price, hour, day = state
        hour = int(hour) - 1
        self._update_running_average(hour, price)
        price_low_to_high_indexes = sorted(range(len(self.means)), key=lambda i: self.means[i])
        # Decision logic for buying
        if storage_level <= 160 and \
        hour in price_low_to_high_indexes[:12 + self.number_of_sales]:
            self._update_purchase_history(price)
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
        self.amount_of_prices = amount_of_prices
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
        if fraction > self.threshold:
            return -1
        if fraction * -1 > self.threshold:
            return 1
        return 0 
        

class TabularQActor(Actor):
    def __init__(self, environment_test, environment_train=None,
                 amount_of_prices: int=161, threshold: float=0.1, alpha=0.005,
                 gamma=0.9, starting_epsilon=0.9, epsilon_decay_rate=0.9995,
                 min_epsilon=0.1, num_episodes=5000,
                 profit_calculation_window=48, Q_init_value=30,
                 minimum_profit=0.3, reward_param=20):
        self.sell_times = 0
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = starting_epsilon
        self.epsilon_decay_rate = epsilon_decay_rate
        self.min_epsilon = min_epsilon
        self.num_episodes = num_episodes
        self.price_bins = [-2, -0.15, 0.25] 
        self.num_hours = 24
        self.actions = [-1, 0, 1]  # Discrete actions
        #self.max_steps = len(environment_train.timestamps) * 24 - 1
        self.storage_bins = [0, 0.4, 0.95]
        self.profit_bools = [True, False]
        self.Q_init_value = Q_init_value
        self.Q = np.full((len(self.price_bins),
                          self.num_hours,
                          len(self.storage_bins),
                          len(self.profit_bools),
                          len(self.actions)),
                          self.Q_init_value)
        
        self.environment_train = environment_train
        self.environment_test = environment_test

        self.current_moving_average : float = 0
        self.current_moving_average_val : float = 0
        self.price_queue = []
        self.price_queue_val = []
        self.amount_of_prices = amount_of_prices
        self.threshold = threshold
        self.last_purchases = []
        self.profit_calculation_window = profit_calculation_window
        self.minimum_profit = minimum_profit
        self.all_profit_margins = []
        self.reward_param = reward_param

    def visualize_trajectory(self, states, from_indx_, to_indx_, nmbr_days_in_fig=3, spam_protection=True):
        end = to_indx_ if to_indx_ is None else to_indx_*24
        window = states[from_indx_*24:end]
        if spam_protection:
                window = window[-6*24:]
        for first_state in range(0, len(window), nmbr_days_in_fig*24):
            # Update the last 24 hours of data
            self.fig, self.ax1 = plt.subplots(figsize=(nmbr_days_in_fig*4, 8))
            self.ax2 = self.ax1.twinx()
            self.plot_prices, hours, days, self.plot_storages, self.plot_actions, self.running_average_plot, self.reward_plot = list(zip(*window[first_state:first_state+nmbr_days_in_fig*24]))

            self.time_labels = []
            for hour, day in zip(hours, days):
                if int(hour) == 1:
                    self.time_labels.append(f"{o_suffix.ordinal(int(day)+1)} day")
                else:
                    self.time_labels.append(str(int(hour)))

            message = f"s: {int(days[0])} to {int(days[-1])}"
            self.trajectory_plotter(message, (0.5, 0.018), 0.2)
        plt.show()

    def trajectory_plotter(self, message, bbox_to_anchor, bottom_space):
        self.ax1.bar(range(len(self.plot_storages)), self.plot_storages, color="orange", alpha=0.6, label="Storage Level")
        action_colors = {
            -1: 'darkred',
            0: 'purple',
            1: 'darkblue'
        }
        action_labels = {
            -1: "Sell: -1",
            0: "No Action",
            1: "Buy: 1"
        }
        for a, color in action_colors.items():
            self.ax1.hlines(0, 0, 0, colors=color, linewidth=4, label=action_labels[a])
        for i, (s, a) in enumerate(zip(self.plot_storages, self.plot_actions)):
            if a == 0:
                self.ax1.hlines(s, i - 0.4, i + 0.4, colors=action_colors[a], linewidth=4)
            elif a in action_colors:
                self.ax1.arrow(i, s, 0, 10 * a, color=action_colors[a], head_width=0.2, head_length=2, length_includes_head=True, linewidth=2)
        self.ax1.plot(range(len(self.plot_prices)), self.plot_prices, 'o', markerfacecolor="white", markeredgecolor="black", markeredgewidth=2, label="Price")
        self.ax1.plot(range(len(self.running_average_plot)), self.running_average_plot, '--', color="black", label="Running Average")
        self.ax2.plot(range(len(self.reward_plot)), self.reward_plot, '-', color='green', label='Reward')
        self.ax1.set_title(f"Tabular Q learning trajectory, day{message}")
        self.ax1.set_xlabel("Time (hours)")
        self.ax1.set_ylabel("Price & Storage Level")
        self.ax2.set_ylabel("Reward")
        self.ax1.set_xticks(range(len(self.time_labels)))
        self.ax1.set_xticklabels(self.time_labels, rotation=60, ha="right", fontsize=9)
        handles_ax1, labels_ax1 = self.ax1.get_legend_handles_labels()
        handles_ax2, labels_ax2 = self.ax2.get_legend_handles_labels()
        handles = handles_ax1 + handles_ax2
        labels = labels_ax1 + labels_ax2
        self.fig.legend(handles, labels, loc="lower center", bbox_to_anchor=bbox_to_anchor, ncol=len(labels), frameon=True)
        plt.tight_layout()
        plt.subplots_adjust(bottom=bottom_space)
    
    def visualize_q_values(self, color_mapping_dim: str):
        dims = {
            'price_bins': 0,
            'hours': 1,
            'storage': 2,
            'extra_dim': 3,
            'action': 4
        }
        action_labels = {
            0: "Sell: -1",
            1: "No Action", 
            2: "Buy: 1" 
        }
        store_bin_labels = [s for s in self.storage_bins]
        store_bin_labels.append("up")
        storage_labels = {
            i: f"{e} - {store_bin_labels[i+1]}" for i, e, in enumerate(store_bin_labels) if type(e) != str
        }
        if color_mapping_dim not in ['storage', 'action']:
            raise ValueError("color_mapping_dim must be one of 'storage' or 'action'")
        color_dim_index = dims[color_mapping_dim]
        other_dim = 'storage' if color_mapping_dim == 'action' else 'action'
        other_dim_index = dims[other_dim]
        extra_dim_index = dims['extra_dim']
        color_bins = self.Q.shape[color_dim_index]
        other_bins = self.Q.shape[other_dim_index]
        extra_bins = self.Q.shape[extra_dim_index]
        fig = plt.figure(figsize=(18, 12)) 
        rows, cols = 2, 3
        grid_positions = [(row, col) for row in range(rows) for col in range(cols)]
        y_ticks = [str(p) for p in self.price_bins]

        for other_bin in range(other_bins):
            for extra_bin in range(extra_bins):
                row, col = grid_positions[other_bin * extra_bins + extra_bin]
                ax = fig.add_subplot(rows, cols, row * cols + col + 1, projection='3d')
                surfaces = []

                for color_bin in range(color_bins):
                    if color_mapping_dim == 'storage':
                        data_slice = self.Q[:, :, color_bin, extra_bin, other_bin]
                    else:
                        data_slice = self.Q[:, :, other_bin, extra_bin, color_bin]
                    max_value = np.max(data_slice)
                    min_value = np.min(data_slice)
                    surfaces.append((max_value, min_value, color_bin, data_slice))
                surfaces.sort(key=lambda x: (x[0], x[1])) 

                for _, _, color_bin, data_slice in surfaces:
                    price_bins, hours = np.meshgrid(
                        np.arange(self.Q.shape[dims['price_bins']]),
                        np.arange(self.Q.shape[dims['hours']])
                    )
                    norm = plt.Normalize(vmin=0, vmax=color_bins - 1)
                    cmap = get_cmap('coolwarm')
                    color_value = norm(color_bin)
                    facecolors = np.empty(data_slice.T.shape + (4,), dtype=float)
                    facecolors[..., :] = cmap(color_value)
                    ax.plot_surface(
                        hours, price_bins, data_slice.T, facecolors=facecolors, shade=False, alpha=0.4, rstride=1, cstride=1
                    )
                if other_dim == 'action':
                    title_label_other = action_labels.get(other_bin, f'Action {other_bin}')
                else:
                    title_label_other = storage_labels.get(other_bin, f'Storage {other_bin}')
                
                title_label_extra = f'Profit: {self.profit_bools[extra_bin]}'
                ax.set_title(f'{other_dim.capitalize()}: {title_label_other}, {title_label_extra}', pad=20)
                ax.set_xlabel('Hours')
                ax.set_ylabel('Price Bins')
                ax.set_zlabel('Q-Values')
                ytick_positions = np.arange(len(y_ticks))
                ax.set_yticks(ytick_positions)
                ax.set_yticklabels(y_ticks)
        norm = plt.Normalize(vmin=0, vmax=color_bins - 1)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        labels = action_labels if color_mapping_dim == 'action' else storage_labels
        ticks = np.arange(color_bins)
        cbar = fig.colorbar(sm, ax=fig.axes, orientation='vertical', fraction=0.025, pad=0.1, ticks=ticks)
        cbar.ax.set_yticklabels([labels.get(i, f"{color_mapping_dim.capitalize()} {i}") for i in ticks])
        cbar.set_label(f'{color_mapping_dim.capitalize()} Values', rotation=90, labelpad=10)
        plt.subplots_adjust(left=0.05, right=0.8, wspace=0.5, hspace=0.5)  # Added space between subplots
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
            
    def calc_storage_fraction(self, current_storage, current_hour):
        current_hour = int(current_hour)
        force_buying_vals = {
            1: 0, 2: 0, 3: 0, 4: 0, 5: 0,
            6: 0, 7: 0, 8: 0, 9: 0, 10: 0,
            11: 0, 12: 0, 13: 10, 14: 20, 15: 30,
            16: 40, 17: 50, 18: 60, 19: 70, 20: 80,
            21: 90, 22: 100, 23: 110, 24: 120}
        max_possible_storage_vals = {
            1: 50, 2: 60, 3: 70, 4: 80, 5: 90,
            6: 100, 7: 110, 8: 120, 9: 130, 10: 140,
            11: 150, 12: 160, 13: 170, 14: 170, 15: 170,
            16: 170, 17: 170, 18: 170, 19: 170, 20: 170,
            21: 170, 22: 170, 23: 170, 24: 170}
        return (current_storage - force_buying_vals[current_hour]) / (max_possible_storage_vals[current_hour] - force_buying_vals[current_hour])
    
    def determine_profitability(self, current_price):
        most_expensive_purchase = max(self.last_purchases, default=0)
        potential_profit = ((current_price * 0.8) - most_expensive_purchase) / most_expensive_purchase if most_expensive_purchase != 0 else 0
        if potential_profit > self.minimum_profit:
            return True
        else:
            return False

    def train(self):
        print('Training the tabular Q-learning agent...')
        validation_rewards = []
        external_rewards = []
        rng = default_rng()

        for episode in range(self.num_episodes):
            state = self.environment_train.reset()
            for t in range(len(environment_train.timestamps) * 24 - 1):
                # Choose action using epsilon-greedy policy
                storage, price, hour, day = state
                fraction = self.calculate_fraction(price)
                storage_fraction = self.calc_storage_fraction(storage, hour)
                self.updateAverage(price)
                profitability = self.determine_profitability(price)
                profit_index = self.profit_bools.index(profitability)

                price_bin_index = np.digitize(fraction, self.price_bins) - 1
                hour_index = int(hour-1)
                if hour_index == 0:
                    self.sell_times = 0
                storage_index = np.digitize(storage_fraction, self.storage_bins) - 1
                if rng.random() < self.epsilon:
                    action_idx = rng.integers(len(self.actions)) # Explore
                else:
                    action_idx = np.argmax(self.Q[price_bin_index, hour_index, storage_index, profit_index, :]) # Exploit
                action = self.actions[action_idx]
                if action < 0:
                    self.sell_times += 1
                elif action > 0:
                    self.last_purchases.append(price)
                    if len(self.last_purchases) == self.profit_calculation_window + 1:
                        self.last_purchases.pop(0)

                next_state, ext_reward, terminated = self.environment_train.step(action)
                next_storage, next_price, next_hour, _ = next_state
                next_fraction = self.calculate_fraction(next_price)
                next_storage_fraction = self.calc_storage_fraction(next_storage, next_hour)
                next_profitability = self.determine_profitability(next_price)
                next_profit_index = self.profit_bools.index(next_profitability)
                next_price_bin_index = np.digitize(next_fraction, self.price_bins) - 1
                next_hour_index = int(next_hour-1)
                next_storage_index = np.digitize(next_storage_fraction, self.storage_bins) - 1

                reward = self.calculate_reward(action,
                                               self.price_bins[price_bin_index],
                                               self.storage_bins[storage_index],
                                               price,
                                               hour,
                                               self.profit_bools[profit_index],
                                               ext_reward)

                best_next_action = np.argmax(self.Q[next_price_bin_index, next_hour_index, next_storage_index, next_profit_index, :])
                self.Q[price_bin_index,
                       hour_index,
                       storage_index,
                       profit_index,
                       action_idx] += self.alpha * (
                                        reward + self.gamma * self.Q[next_price_bin_index,
                                                                    next_hour_index,
                                                                    next_storage_index,
                                                                    next_profit_index,
                                                                    best_next_action] - self.Q[price_bin_index,
                                                                                                hour_index,
                                                                                                storage_index,
                                                                                                profit_index,
                                                                                                action_idx])
                state = next_state
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
        with open("final_tab_q_val_rewards_v2.csv", "w") as f:
            writer = csv.writer(f)
            writer.writerow(["Validation Rewards", "External Rewards"])
            for internal, external in zip(validation_rewards, external_rewards):
                writer.writerow([internal, external])
        np.save('final_tab_q_Q_tensor.npy', self.Q)
        #self.visualize_q_values('storage')
        #self.visualize_q_values('action')
        
    def act(self, state):
        storage, price, hour, _ = state
        fraction = self.calculate_fraction(price, train=False)
        storage_fraction = self.calc_storage_fraction(storage, hour)
        self.updateAverage(price, train=False)
        price_bin_index = np.digitize(fraction, self.price_bins) - 1
        profitability = self.determine_profitability(price)
        profit_index = self.profit_bools.index(profitability)
        hour_index = int(hour-1)
        if hour_index == 0:
            self.sell_times = 0
        storage_index = np.digitize(storage_fraction, self.storage_bins) - 1
        best_action_index = np.argmax(self.Q[price_bin_index, hour_index, storage_index, profit_index, :])
        best_action = self.actions[best_action_index]
        if best_action < 0:
            self.sell_times += 1
        elif best_action > 0:
            self.last_purchases.append(price)
            if len(self.last_purchases) == self.profit_calculation_window + 1:
                self.last_purchases.pop(0)
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

    def calculate_reward(self,
                         action,
                         price_difference,
                         storage_level,
                         current_price,
                         current_hour,
                         profit_bool,
                         env_reward):
            
        if action < 0: # Selling
            if price_difference >= self.price_bins[-1] and storage_level >= self.storage_bins[1] and profit_bool:
                most_expensive_purchase = max(self.last_purchases, default=0)
                potential_profit = ((current_price * 0.8) - most_expensive_purchase) / most_expensive_purchase if most_expensive_purchase != 0 else 0
                return action * potential_profit * -100
            else:
                return -50
        
        elif action == 0: # Do nothing
            if price_difference >= self.price_bins[-1] and not profit_bool:
                return 5
            return 0

        elif action > 0: # Buying
            if current_hour <= 12:
                buy_when_low_storage_reward = max(0, (0.4 - storage_level)/0.4) - price_difference
            elif storage_level == self.storage_bins[-1] and current_hour > 12:
                return -10
            else:
                buy_when_low_storage_reward = 0
            return action * price_difference * -10 + self.reward_param * buy_when_low_storage_reward


    def val(self):
        aggregate_reward = 0
        aggregate_test_reward = 0
        terminated = False
        state = self.environment_test.reset()
        while not terminated:
            action = self.act(state)
            next_state, reward, terminated = self.environment_test.step(action)
            state = next_state
            storage, price, hour, day = state
            fraction = self.calculate_fraction(price, train=False)
            storage_fraction = self.calc_storage_fraction(storage, hour)
            profitability = self.determine_profitability(price)
            price_bin_index = np.digitize(fraction, self.price_bins) - 1
            storage_index = np.digitize(storage_fraction, self.storage_bins) - 1
            profit_index = self.profit_bools.index(profitability)

            aggregate_reward += self.calculate_reward(action,
                                                      self.price_bins[price_bin_index],
                                                      self.storage_bins[storage_index],
                                                      price,
                                                      hour,
                                                      self.profit_bools[profit_index],
                                                      reward)
            aggregate_test_reward += reward

        return aggregate_reward, aggregate_test_reward