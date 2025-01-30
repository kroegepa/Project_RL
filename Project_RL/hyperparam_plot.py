import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

def visualize_hyperparam_search(csv_file: str):
    """
    Visualizes the hyperparameter search results using 3D surface plots.

    Each subplot represents a unique combination of (reward_param, gamma),
    with:
      - X-axis: last_purchases_window
      - Y-axis: alpha
      - Z-axis: total validation reward
      - Color: profitability_margins (mapped to a colormap from cold to hot)

    :param csv_file: Path to the CSV file containing hyperparameter search results.
    """
    # Load the data
    df = pd.read_csv(csv_file)
    
    # Extract unique values for reward_param and gamma to create subplots
    reward_params = sorted(df['reward_param'].unique())
    gammas = sorted(df['gamma'].unique())
    
    # Set up 2x2 grid for subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), subplot_kw={'projection': '3d'})
    
    # Define colormap for profitability_margins
    unique_margins = sorted(df['profitability_margins'].unique())
    norm = plt.Normalize(vmin=min(unique_margins), vmax=max(unique_margins))
    cmap = cm.get_cmap("coolwarm")
    
    # Iterate over reward_param and gamma to generate subplots
    for i, reward_param in enumerate(reward_params):
        for j, gamma in enumerate(gammas):
            ax = axes[i, j]
            
            # Filter data for current (reward_param, gamma) combination
            subset = df[(df['reward_param'] == reward_param) & (df['gamma'] == gamma)]
            
            # Generate surface plot
            for margin in unique_margins:
                data = subset[subset['profitability_margins'] == margin]
                if not data.empty:
                    X_vals = sorted(data['last_purchases_window'].unique())
                    Y_vals = sorted(data['alpha'].unique())
                    X, Y = np.meshgrid(X_vals, Y_vals)
                    Z = np.array([[data[(data['last_purchases_window'] == x) & (data['alpha'] == y)]['total validation reward'].mean() 
                                   for x in X[0]] for y in Y[:, 0]])
                    
                    color_value = cmap(norm(margin))
                    facecolors = np.full(X.shape + (4,), color_value)  # Ensure facecolors has the correct shape
                    ax.plot_surface(X, Y, Z, facecolors=facecolors, shade=False, edgecolor='k', alpha=0.6)
            
            # Set labels and title
            ax.set_xlabel('Last Purchases Window')
            ax.set_ylabel('Alpha')
            ax.set_zlabel('Total Validation Reward')
            ax.set_title(f"Reward Param: {reward_param}, Gamma: {gamma}")
            
            # Adjust axis limits for better readability
            ax.set_xticks(sorted(df['last_purchases_window'].unique()))
            ax.set_yticks(sorted(df['alpha'].unique()))
            
    # Create the colorbar legend for profitability_margins
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=fig.axes, orientation='vertical', fraction=0.025, pad=0.1)
    cbar.set_label("Profitability Margins", rotation=90, labelpad=10)
    plt.subplots_adjust(left=0.1, right=0.7, hspace=0.6, wspace=0.6)

    plt.show()



df = pd.read_csv("hyper_param_tab_q_val_test.csv")
df_sorted = df.sort_values(by="total validation reward", ascending=False)  # Change to True for ascending order
top_5 = df_sorted.head(10)
print(top_5)


visualize_hyperparam_search("hyper_param_tab_q_val_test.csv")
