from env import DataCenterEnv
import numpy as np
import argparse
import actor

args = argparse.ArgumentParser()
args.add_argument('--path_data', type=str, default='train.xlsx')
args.add_argument('--path_q', type=str, default='Q.npy')
args = args.parse_args()

np.set_printoptions(suppress=True, precision=2)
path_to_dataset = args.path_data
path_to_Q = args.path_q
environment = DataCenterEnv(path_to_dataset)

Q = np.load(path_to_Q)
agent = actor.TabularQActor(Q=Q)

aggregate_reward = 0
terminated = False
state = environment.observation()
while not terminated:
    # agent is your own imported agent class
    action = agent.act(state)
    # next_state is given as: [storage_level, price, hour, day]
    next_state, reward, terminated = environment.step(action)
    state = next_state
    aggregate_reward += reward
    print("Action:", action)
    print("Next state:", next_state)
    print("Reward:", reward)

print('Total reward:', aggregate_reward)
