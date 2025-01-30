from env import DataCenterEnv
import numpy as np
import argparse
import actor
import csv
from itertools import product


args = argparse.ArgumentParser()
args.add_argument('--path_train', type=str, default='Project_RL/train.xlsx')
args.add_argument('--path_test', type=str, default='Project_RL/validate.xlsx')
args.add_argument('--actor_type', type=str, default='tabular_q')
args = args.parse_args()

np.set_printoptions(suppress=True, precision=2)
path_to_dataset_train = args.path_train
path_to_dataset_test = args.path_test
actor_type = args.actor_type

environment_train = DataCenterEnv(path_to_dataset_train)
environment_test = DataCenterEnv(path_to_dataset_test)


aggregate_reward = 0
terminated = False
#state = environment_test.observation()
if actor_type == 'uniform_baseline':
    agent = actor.UniformBuyActor()
elif actor_type == 'tabular_q':
    agent = actor.TabularQActor(environment_test=environment_test, 
                                environment_train=environment_train,
                                num_episodes=1500, starting_epsilon=1,
                                min_epsilon=0.1, epsilon_decay_rate=0.998,
                                alpha=0.005, gamma=0.9, minimum_profit=0.25,
                                profit_calculation_window=48, reward_param=20)
    agent.train()
elif actor_type == 'SMA':
    agent = actor.SimpleMovingAverageActor()
elif actor_type == 'EMA':
    agent = actor.ExponentialMovingAverage(number_of_sales=2)

average_filled = 0
amount_of_days = 0
state = environment_test.reset()
with open("trajectory_file_.csv", "w") as file:
    writer = csv.writer(file)
    writer.writerow(["price", "hour", "day", "storage", "action", "moving_average", "plotting_reward"])
    while not terminated:
        # agent is our own imported agent class
        action = agent.act(state)
        next_state, reward, terminated = environment_test.step(action)
        if actor_type == 'tabular_q':
            storage,price,hour,day = state
            fraction = agent.calculate_fraction(price, train=False)
            price_bin_index = np.digitize(fraction, agent.price_bins) - 1
            storage_fraction = agent.calc_storage_fraction(storage, hour)
            storage_index = np.digitize(storage_fraction, agent.storage_bins) - 1
            profitability = agent.determine_profitability(price)
            profit_index = agent.profit_bools.index(profitability)
            plotting_reward = agent.calculate_reward(action,agent.price_bins[price_bin_index],agent.storage_bins[storage_index],price,hour,agent.profit_bools[profit_index],reward)
            writer.writerow([price, hour, day, storage, action, agent.current_moving_average_val, plotting_reward])
        
        # next_state is given as: [storage_level, price, hour, day]
        state = next_state
        aggregate_reward += reward
        storage,price,hour,day = state
        if hour == 24:
            average_filled += storage
            amount_of_days += 1
        
print(f'Total reward for actor {actor_type}: {round(aggregate_reward):,}')
print(f'Reward per year for actor {actor_type}: {round(aggregate_reward / 2):,}') # devide by 3 if run on the train set

