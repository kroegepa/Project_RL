import actor
import numpy as np
from env import DataCenterEnv

addon = "test_p0.1_w5_g0.9_a0.01_r1_"

trajectory_path = "tab_q_val_trajectory.csv"
validation_reward_path = "tab_q_val_rewards.csv"
validation_Q_tensor = "tab_q_Q_tensor.npy"

trajectory_data = np.loadtxt(addon + trajectory_path, delimiter=",", skiprows=1)
validation_reward_data = np.loadtxt(addon + validation_reward_path, delimiter=",", skiprows=1)
Qs = np.load(addon + validation_Q_tensor)

environment_train = DataCenterEnv('train.xlsx')
environment_test = DataCenterEnv('validate.xlsx')
agent = actor.TabularQActor(environment_train, environment_test)
agent.Q = Qs

agent.visualize_trajectory(trajectory_data, 300, 330, nmbr_days_in_fig=5, spam_protection=False)
agent.plot_rewards_during_training(validation_reward_data)
agent.visualize_q_values('storage')
agent.visualize_q_values('action')
