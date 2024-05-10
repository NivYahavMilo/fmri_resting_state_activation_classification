import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import seaborn as sns


def plot_roi_temporal_windows_dynamic(data: dict, mode: str, roi: str = None):
    if len(data.keys()) == 14:
        roi_data = data
        data = {roi: roi_data}

    sns.set_style("darkgrid")  # Set Seaborn grid style
    fig, ax = plt.subplots(figsize=(12, 8))

    for region, dynamics in data.items():
        windows_range = list(dynamics.keys())
        avg_values = [dynamics[key]['avg'] for key in windows_range]
        std_values = [dynamics[key]['std'] for key in windows_range]

        # Plot average for each subkey
        ax.plot(windows_range, avg_values, label=region)

        # Fill between average +/- std
        ax.fill_between(windows_range, np.array(avg_values) - np.array(std_values),
                        np.array(avg_values) + np.array(std_values), alpha=0.3)

    # Customize the plot
    ax.set_xlabel('Temporal Windows')
    ax.set_ylabel('Accuracy')
    ax.set_title(f'Rois {mode.title()} Dynamics with Average Values and Standard Deviation')
    ax.legend()

    # Rotate x-axis labels for better readability
    plt.xticks(np.arange(len(windows_range)), windows_range, rotation=45)
    # plt.ylim(0, 1)

    # Show the plot
    plt.show()


if __name__ == '__main__':
    distances_results = pd.read_pickle("rois_distances_results_avg.pkl")
    plot_roi_temporal_windows_dynamic(data=distances_results, mode='distances')
    activations_results = pd.read_pickle("rois_activations_results_avg_3.pkl")
    plot_roi_temporal_windows_dynamic(data=activations_results, mode='activations')
