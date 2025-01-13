import env
import actor

# Parameters
alpha = 0.1  # Learning rate
gamma = 0.9  # Discount factor
epsilon = 0.1  # Exploration rate
num_bins_price = 21  # Discretization bins for price
num_actions = 5  # Discrete actions

path_to_data = 'train.xlsx'
environment = env.DataCenterEnv(path_to_data)
print(max(environment.price_values))

# Initialize Q-table
#Q = np.zeros((num_states, num_actions))

# Training loop
train = False
if train:
    for episode in range(num_episodes):
        state = initial_state  # Start at the initial state
        for t in range(max_steps):
            # Choose action using epsilon-greedy policy
            if np.random.rand() < epsilon:
                action = np.random.choice(num_actions)  # Explore
            else:
                action = np.argmax(Q[state, :])  # Exploit

            # Take action, observe reward and next state
            next_state, reward = environment.step(state, action)

            # Update Q-value
            best_next_action = np.argmax(Q[next_state, :])
            Q[state, action] += alpha * (reward + gamma * Q[next_state, best_next_action] - Q[state, action])

            # Transition to next state
            state = next_state
