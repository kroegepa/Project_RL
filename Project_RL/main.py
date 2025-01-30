from env import DataCenterEnv
import numpy as np
import argparse
import actor
import csv
from itertools import product


args = argparse.ArgumentParser()
args.add_argument('--path_train', type=str, default='train.xlsx')
args.add_argument('--path_test', type=str, default='validate.xlsx')
args.add_argument('--actor_type', type=str, default='tabular_q')
args = args.parse_args()

np.set_printoptions(suppress=True, precision=2)
path_to_dataset_train = args.path_train
path_to_dataset_test = args.path_test
actor_type = args.actor_type

environment_train = DataCenterEnv(path_to_dataset_train)
environment_test = DataCenterEnv(path_to_dataset_test)


profitability_margins = [0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
last_purchases_window = [24, 36, 48, 60, 72, 84, 96]
gamma = [0.85, 0.9, 0.95, 0.99]
alpha = [0.05, 0.01, 0.005, 0.001]
reward_param = [10, 20, 30, 40]


for p, w, g, a, r in product(
    profitability_margins, last_purchases_window, gamma, alpha, reward_param
):
    test_name = f"test_p{p}_w{w}_g{g}_a{a}_r{r}"
    print(f"Running {test_name}")


    aggregate_reward = 0
    terminated = False
    #state = environment_test.observation()
    if actor_type == 'uniform_baseline':
        agent = actor.UniformBuyActor()
    elif actor_type == 'tabular_q':
        agent = actor.TabularQActor(environment_train, environment_test,
                                    num_episodes=300, starting_epsilon=0.4,
                                    min_epsilon=0.1, epsilon_decay_rate=0.999,
                                    alpha=0.01)
        agent.train(test_name)
    elif actor_type == 'SMA':
        agent = actor.SimpleMovingAverageActor()
    elif actor_type == 'EMA':
        agent = actor.ExponentialMovingAverage(number_of_sales=2)

    spam = False
    average_filled = 0
    amount_of_days = 0
    state = environment_test.reset()
    with open(f"{test_name}_tab_q_val_trajectory.csv", "w") as file:
        writer = csv.writer(file)
        writer.writerow(["price", "hour", "day", "storage", "action", "moving_average", "plotting_reward"])
        while not terminated:
            # agent is your own imported agent class
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
            #action = np.random.uniform(-1, 1)
            # next_state is given as: [storage_level, price, hour, day]
            state = next_state
            aggregate_reward += reward
            storage,price,hour,day = state
            if hour == 24:
                average_filled += storage
                amount_of_days += 1
            if spam:
                print("Action:", action)
                print("Next state:", next_state)
                print("Reward:", reward)
                if actor_type == 'tabular_q':
                    fraction = agent.calculate_fraction(price)
                    price_bin_index = np.digitize(fraction, agent.price_bins) - 1
                    storage_index = np.digitize(storage, agent.storage_bins) - 1
                    actor_reward_negative = agent.calculate_reward(-1,agent.price_bins[price_bin_index],agent.storage_bins[storage_index])
                    actor_reward_positve = agent.calculate_reward(1,agent.price_bins[price_bin_index],agent.storage_bins[storage_index])
                    print("Agent reward negative:", actor_reward_negative)
                    print("Agent reward postive:", actor_reward_positve)
    print(f'Total reward for actor {actor_type}: {round(aggregate_reward):,}')
    print(f'Reward per year for actor {actor_type}: {round(aggregate_reward / 2):,}') # devide by 3 if run on the train set
    #print(average_filled/amount_of_days)
    #print(agent.Q)
    #print(np.shape(agent.Q))
    break