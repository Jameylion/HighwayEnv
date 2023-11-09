import pandas as pd
import matplotlib.pyplot as plt

def plot_csv_to_eps(file1, file2, file3, output_file, title):
    # Read CSV files into pandas dataframes
    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)
    df3 = pd.read_csv(file3)

    # Create subplots
    fig, axs = plt.subplots()

    # Plot data from each CSV file
    axs.plot(df1['Step'], df1['Value'], label='PPO')
    axs.plot(df2['Step'], df2['Value'], label='TRPO')
    axs.plot(df3['Step'], df3['Value'], label='DQN')

    # Set labels and title
    axs.set_xlabel('Step')
    axs.set_ylabel('Value')
    axs.set_title(title)

    # Add legend
    axs.legend()

    # Save the plot as EPS file
    plt.savefig(output_file, format='eps')

folder = "C:/Users/Admin/downloads"
plot_csv_to_eps(folder+ "/run-merge_in_ppo_PPO_2-tag-rollout_ep_len_mean.csv",
                 folder+"/run-merge_in_TRPO_TRPO_2-tag-rollout_ep_len_mean.csv",
                   folder+"/run-merge_in_dqn_DQN_7-tag-rollout_ep_len_mean.csv", 
                   "D:/OneDrive/Studie/Embedded_Systems/Master_year_2/FAIP_IFEEMCS520100/Project/ep_len_graph.eps", "Mean Episode Length")
plot_csv_to_eps(folder+ "/run-merge_in_ppo_PPO_2-tag-rollout_ep_rew_mean.csv",
                 folder+"/run-merge_in_TRPO_TRPO_2-tag-rollout_ep_rew_mean.csv",
                   folder+"/run-merge_in_dqn_DQN_7-tag-rollout_ep_rew_mean.csv", 
                   "D:/OneDrive/Studie/Embedded_Systems/Master_year_2/FAIP_IFEEMCS520100/Project/ep_rew_graph.eps", "Mean Episode Reward")