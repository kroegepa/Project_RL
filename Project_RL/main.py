from env import DataCenterEnv
import numpy as np
import argparse
import actor

args = argparse.ArgumentParser()
args.add_argument('--path_train', type=str, default='train.xlsx')
args.add_argument('--path_test', type=str, default='validate.xlsx')
args.add_argument('--actor_type', type=str, default='uniform_baseline')
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
    agent = actor.TabularQActor(environment_train, environment_test,num_episodes=1000, epsilon_decay_rate=0.995)
    agent.train()
elif actor_type == 'SMA':
    agent = actor.SimpleMovingAverageActor()
elif actor_type == 'EMA':
    agent = actor.ExponentialMovingAverage(number_of_sales=2)

average_filled = 0
amount_of_days = 0
state = environment_test.reset()
while not terminated:
    # agent is your own imported agent class
    action = agent.act(state)
    #action = np.random.uniform(-1, 1)
    # next_state is given as: [storage_level, price, hour, day]
    next_state, reward, terminated = environment_test.step(action)
    state = next_state
    aggregate_reward += reward
    storage,price,hour,_ = state
    if actor_type == 'tabular_q':
        fraction = agent.calculate_fraction(price)
        price_bin_index = np.digitize(fraction, agent.bins) - 1
        storage_index = np.digitize(storage, agent.storage_bins) - 1
        actor_reward_negative = agent.calculate_reward(-1,agent.bins[price_bin_index],agent.storage_bins[storage_index])
        actor_reward_positve = agent.calculate_reward(1,agent.bins[price_bin_index],agent.storage_bins[storage_index])
    if hour == 24:
        average_filled += storage
        amount_of_days += 1
    print("Action:", action)
    print("Next state:", next_state)
    print("Reward:", reward)
    if actor_type == 'tabular_q':
        print("Agent reward negative:", actor_reward_negative)
        print("Agent reward postive:", actor_reward_positve)
print(f'Total reward for actor {actor_type}: {aggregate_reward}')
print(f'Reward per year for actor {actor_type}: {aggregate_reward / 2}') # devide by 3 if run on the train set
print(average_filled/amount_of_days)
print(agent.Q)
print(np.shape(agent.Q))