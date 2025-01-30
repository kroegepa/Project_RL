from env import DataCenterEnv
import numpy as np
import argparse
import actor
import csv
from itertools import product
import multiprocessing


def run_experiment(params):
    """Runs a single experiment with given hyperparameters."""
    p, w, g, a, r = params
    test_name = f"test_p{p}_w{w}_g{g}_a{a}_r{r}"
    #print(f"Running {test_name}")

    environment_train = DataCenterEnv(args.path_train)
    environment_test = DataCenterEnv(args.path_test)

    aggregate_reward = 0
    terminated = False

    if args.actor_type == 'uniform_baseline':
        agent = actor.UniformBuyActor()
    elif args.actor_type == 'tabular_q':
        agent = actor.TabularQActor(
            environment_train, environment_test, num_episodes=1500,
            starting_epsilon=1, min_epsilon=0.1, epsilon_decay_rate=0.9995,
            minimum_profit=p, profit_calculation_window=w, gamma=g,
            alpha=a, reward_param=r
        )
        agent.train(test_name)
    elif args.actor_type == 'SMA':
        agent = actor.SimpleMovingAverageActor()
    elif args.actor_type == 'EMA':
        agent = actor.ExponentialMovingAverage(number_of_sales=2)

    state = environment_test.reset()
    with open(f"{test_name}_tab_q_val_trajectory.csv", "w") as file:
        writer = csv.writer(file)
        writer.writerow(["price", "hour", "day", "storage", "action", "moving_average", "plotting_reward"])
        while not terminated:
            action = agent.act(state)
            next_state, reward, terminated = environment_test.step(action)
            if args.actor_type == 'tabular_q':
                storage, price, hour, day = state
                fraction = agent.calculate_fraction(price, train=False)
                price_bin_index = np.digitize(fraction, agent.price_bins) - 1
                storage_fraction = agent.calc_storage_fraction(storage, hour)
                storage_index = np.digitize(storage_fraction, agent.storage_bins) - 1
                profitability = agent.determine_profitability(price)
                profit_index = agent.profit_bools.index(profitability)
                plotting_reward = agent.calculate_reward(action, agent.price_bins[price_bin_index], agent.storage_bins[storage_index], price, hour, agent.profit_bools[profit_index], reward)
                writer.writerow([price, hour, day, storage, action, agent.current_moving_average_val, plotting_reward])
            state = next_state
            aggregate_reward += reward

    #print(f'Total reward for actor {args.actor_type}: {round(aggregate_reward):,}')
    with open("hyper_param_tab_q_val_test.csv", "a") as file:
        writer = csv.writer(file)
        writer.writerow([round(aggregate_reward), p, w, g, a, r])


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument('--path_train', type=str, default='train.xlsx')
    args.add_argument('--path_test', type=str, default='validate.xlsx')
    args.add_argument('--actor_type', type=str, default='tabular_q')
    args = args.parse_args()

    np.set_printoptions(suppress=True, precision=2)

    profitability_margins = [0.25, 0.35, 0.45]
    last_purchases_window = [24, 48, 72, 96]
    gamma = [0.9, 0.99]
    alpha = [0.01, 0.005, 0.001]
    reward_param = [20, 35]

    param_combinations = list(product(profitability_margins, last_purchases_window, gamma, alpha, reward_param))

    with open("hyper_param_tab_q_val_test.csv", "w") as file:
        writer = csv.writer(file)
        writer.writerow(["total validation reward", "profitability_margins", "last_purchases_window", "gamma", "alpha", "reward_param"])

    # Use multiprocessing
    num_workers = min(multiprocessing.cpu_count(), len(param_combinations))  # Use as many CPUs as available
    with multiprocessing.Pool(num_workers) as pool:
        pool.map(run_experiment, param_combinations)
