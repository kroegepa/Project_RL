import actor
import numpy as np
from env import DataCenterEnv


trajectory_path = "tab_q_val_trajectory.csv"
validation_reward_path = "tab_q_val_rewards.csv"

trajectory_data = np.loadtxt(trajectory_path, delimiter=",", skiprows=1)
validation_reward_data = np.loadtxt(validation_reward_path, delimiter=",", skiprows=1)

environment_train = DataCenterEnv('train.xlsx')
environment_test = DataCenterEnv('validate.xlsx')
agent = actor.TabularQActor(environment_train, environment_test)

agent.visualize_trajectory(trajectory_data, 550, 570, nmbr_days_in_fig=5, spam_protection=False)
agent.plot_rewards_during_training(validation_reward_data)

