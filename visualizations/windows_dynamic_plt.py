import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import seaborn as sns


def _plot_roi_temporal_windows_dynamic(data: dict, mode: str, roi: str = None):
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


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def plot_roi_temporal_windows_dynamic(data_normal, data_resting, mode, regions):
    sns.set_style("darkgrid")  # Set Seaborn grid style

    fig, axes = plt.subplots(2, 1, figsize=(12, 16))
    axes = axes.flatten()

    for i, region in enumerate(regions):
        ax = axes[i]

        for state, data in [('Stimulated State', data_normal), ('Resting State', data_resting)]:
            region_data = data.get(region, {})
            windows_range = list(region_data.keys())
            avg_values = [region_data[key]['avg'] for key in windows_range]
            std_values = [region_data[key]['std'] for key in windows_range]

            # Plot average for each subkey
            ax.plot(windows_range, avg_values, label=f"{region} - {state}")

            # Fill between average +/- std
            ax.fill_between(windows_range, np.array(avg_values) - np.array(std_values),
                            np.array(avg_values) + np.array(std_values), alpha=0.3)

        # Customize the plot
        ax.set_xlabel('Temporal Windows')
        ax.set_ylabel('Accuracy')
        title = f'{region} {mode.title()} Dynamics with Average Values and Standard Deviation'
        ax.set_title(title)
        ax.legend()

        # Rotate x-axis labels for better readability
        ax.set_xticks(np.arange(len(windows_range)))
        ax.set_xticklabels(windows_range, rotation=45)
        ax.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(f'figures/final/{title}', dpi=300)
    plt.show()


if __name__ == '__main__':
    distances_results = pd.read_pickle("all_rois_groups_distances_results.pkl")
    distances_resting_state_results = pd.read_pickle("all_rois_groups_distances_results_resting_state.pkl")
    activations_results = pd.read_pickle("all_rois_groups_activations_results.pkl")
    activations_resting_state_results = pd.read_pickle("all_rois_groups_activations_results_resting_state.pkl")

    regions = ['RH_DorsAttn_Post_2', 'LH_SomMot_4']  # Specify your regions of interest

    plot_roi_temporal_windows_dynamic(data_normal=distances_results, data_resting=distances_resting_state_results,
                                      mode='distances', regions=regions)
    plot_roi_temporal_windows_dynamic(data_normal=activations_results, data_resting=activations_resting_state_results,
                                      mode='activations', regions=regions)


