from env import DataCenterEnv
import numpy as np
import argparse
import actor
max_reward = -1000000000
current_i = 0
for i in range(1,240):

    args = argparse.ArgumentParser()
    args.add_argument('--path', type=str, default='train.xlsx')
    args = args.parse_args()

    np.set_printoptions(suppress=True, precision=2)
    path_to_dataset = args.path

    environment = DataCenterEnv(path_to_dataset)

    aggregate_reward = 0
    terminated = False
    state = environment.observation()

    agent = actor.AveragedBuyingActor(amount_of_prices=i,threshold=0.05)
    while not terminated:
        # agent is your own imported agent class
        action = agent.act(state)
        #action = np.random.uniform(-1, 1)
        # next_state is given as: [storage_level, price, hour, day]
        next_state, reward, terminated = environment.step(action)
        
        state = next_state
        aggregate_reward += reward
        #print("Action:", action)
        #print("Next state:", next_state)
        #print("Reward:", reward)
    if max_reward < aggregate_reward:
        max_reward = aggregate_reward
        current_i = i
    print(i)
        



print('Total reward:', aggregate_reward)
print(max_reward)
print(current_i)