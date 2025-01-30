TRAINING THE MODEL:

When training the tabular Q-learning agent, you need to run the file main.py as follows:

$ python Project_RL/main.py

It will train and validate the agent and save the Q-table as a .npy file, the rewards to a .csv file and the trajectory information (for plotting) to a .csv file.

If you want to get the validation results of one of the baselines, use the following line with as actor_type one of: [uniform_baseline, SMA, EMA]

$ python Project_RL/main.py --actor_type {actor_type}

TESTING THE MODEL:

Run the file main_testing.py as follows:

$ python Project_RL/main_testing.py --path_data {path_to_test_data} --path_q Project_RL/Q.npy
