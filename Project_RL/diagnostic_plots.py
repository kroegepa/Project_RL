import actor
import numpy as np
from env import DataCenterEnv

trajectory_data = np.loadtxt("tab_q_val_trajectory_2k.csv", delimiter=",", skiprows=1)
validation_reward_data = np.loadtxt("tab_q_val_rewards.csv", delimiter=",", skiprows=1)

environment_train = DataCenterEnv('train.xlsx')
environment_test = DataCenterEnv('validate.xlsx')
agent = actor.TabularQActor(environment_train, environment_test)

agent.visualize_trajectory(trajectory_data, 724, None, nmbr_days_in_fig=5, spam_protection=False)
agent.plot_rewards_during_training(validation_reward_data)

