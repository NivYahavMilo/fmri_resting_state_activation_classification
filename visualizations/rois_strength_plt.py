import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from static_data.static_data import StaticData

StaticData.inhabit_class_members()


def plot_sorted_activations_single_plot(roi_to_net_map, activation_results, window=None, method='mean'):
    # Dictionary to hold activation values and corresponding networks
    activation_data = []

    # Populate the data for all ROIs
    for roi, network in roi_to_net_map.items():
        roi_activations = activation_results.get(roi, {})

        # Handle window-based data or aggregate method
        if window is None:
            if method == 'mean':
                act_avg_values = [roi_activations[win]['avg'] for win in roi_activations]
                activation_value = sum(act_avg_values) / len(act_avg_values)
            else:
                act_max_values = [roi_activations[win]['avg'] for win in roi_activations]
                activation_value = max(act_max_values)
        elif window == 'last_tr_movie':
            activation_value = roi_activations['avg']

        else:
            activation_value = roi_activations.get(window, {}).get('avg', None)

        # Store only if activation data is available
        if activation_value is not None:
            activation_data.append((roi, activation_value, network))

    # Sort the data by activation values (high to low)
    activation_data.sort(key=lambda x: x[1], reverse=True)

    # Extract the sorted ROI labels, activation values, and network assignments
    sorted_rois = [item[0] for item in activation_data]
    sorted_activation_values = [item[1] for item in activation_data]
    sorted_networks = [item[2] for item in activation_data]

    # Set up colors for the networks
    networks = list(set(sorted_networks))
    palette = sns.color_palette("hsv", len(networks))  # Create distinct colors
    network_colors = {network: palette[i] for i, network in enumerate(networks)}

    # Assign colors to each bar based on its network
    bar_colors = [network_colors[net] for net in sorted_networks]

    # Plot the sorted activations
    fig, ax = plt.subplots(figsize=(20, 10))
    x = np.arange(len(sorted_rois))

    bars = ax.bar(x, sorted_activation_values, color=bar_colors)

    # Add labels and title
    ax.set_xlabel('ROIs', fontsize=15)
    ax.set_ylabel('Activation Values', fontsize=15)
    ax.set_title(f'Sorted Activation Values by ROI (Colored by Network) Window:{window}', fontsize=20)

    # Set tick labels and rotation for clarity
    #ax.set_xticks(x)
    #ax.set_xticklabels(sorted_rois, rotation=90, fontsize=10)

    # Create a custom legend for networks
    legend_patches = [
        plt.Line2D([0], [0], marker='o', color=network_colors[net], label=net, markersize=10, linestyle='')
        for net in networks]
    ax.legend(handles=legend_patches, title="Networks", fontsize=12, title_fontsize=14)

    # Adjust layout and show the plot
    fig.tight_layout()
    fig.savefig(f'rois_strength_plt_{window}_window.png')
    plt.show()


def plot_roi_value(results, method=None, window=None):
    if method not in ['mean', 'max'] and window is None:
        print("Invalid input. Please choose either 'mean' or 'max' as method or specify a window.")
        return

    roi_to_net_map = StaticData.ROI_TO_NETWORK_MAPPING
    network_results = {network: {} for network in set(roi_to_net_map.values())}

    for roi, network in roi_to_net_map.items():
        network_results[network][roi] = results.get(roi, {})

    for network, rois in network_results.items():
        roi_labels = list(rois.keys())
        values = []
        std_devs = []

        if window is None:
            for roi in rois.values():
                if method == 'mean':
                    avg_values = [roi[win]['avg'] for win in roi]
                    values.append(sum(avg_values) / len(avg_values))
                    std_devs.append(sum(roi[win]['std'] for win in roi) / len(roi))
                else:
                    max_values = [roi[win]['avg'] for win in roi]
                    values.append(max(max_values))
                    std_devs.append(max(roi[win]['std'] for win in roi))

            title = f"{method.capitalize()} values for {network} network"
        else:
            for roi in rois.values():
                value_at_window = roi.get(window)
                if value_at_window is None:
                    values.append(None)
                    std_devs.append(None)
                else:
                    values.append(value_at_window['avg'])
                    std_devs.append(value_at_window['std'])
            title = f"Value for {network} network at window '{window}'"

        plt.figure(figsize=(40, 20))
        plt.xticks(rotation=75, fontsize=30)  # Adjust fontsize as needed
        plt.yticks(fontsize=20)
        plt.bar(roi_labels, values, color='skyblue', yerr=std_devs, capsize=5)
        plt.title(title, fontsize=30)
        plt.xlabel('ROIs')
        plt.ylabel('Value')
        # plt.savefig(f"figures/distances/{title}.png")
        plt.show()


def plot_combined_roi_value(activation_results, distance_results, method=None, window=None):
    if method not in ['mean', 'max'] and window is None:
        print("Invalid input. Please choose either 'mean' or 'max' as method or specify a window.")
        return

    roi_to_net_map = StaticData.ROI_TO_NETWORK_MAPPING
    network_results = {network: {} for network in set(roi_to_net_map.values())}

    for roi, network in roi_to_net_map.items():
        network_results[network][roi] = {
            'activations': activation_results.get(roi, {}),
            'distances': distance_results.get(roi, {})
        }

    for network, rois in network_results.items():
        roi_labels = list(rois.keys())
        activation_values = []
        activation_std_devs = []
        distance_values = []
        distance_std_devs = []

        if window is None:
            for roi_data in rois.values():
                activation_data = roi_data['activations']
                distance_data = roi_data['distances']

                if method == 'mean':
                    act_avg_values = [activation_data[win]['avg'] for win in activation_data]
                    dis_avg_values = [distance_data[win]['avg'] for win in distance_data]

                    activation_values.append(sum(act_avg_values) / len(act_avg_values))
                    distance_values.append(sum(dis_avg_values) / len(dis_avg_values))

                    activation_std_devs.append(
                        sum(activation_data[win]['std'] for win in activation_data) / len(activation_data))
                    distance_std_devs.append(
                        sum(distance_data[win]['std'] for win in distance_data) / len(distance_data))
                else:
                    act_max_values = [activation_data[win]['avg'] for win in activation_data]
                    dis_max_values = [distance_data[win]['avg'] for win in distance_data]

                    activation_values.append(max(act_max_values))
                    distance_values.append(max(dis_max_values))

                    activation_std_devs.append(max(activation_data[win]['std'] for win in activation_data))
                    distance_std_devs.append(max(distance_data[win]['std'] for win in distance_data))

            title = f"{method.capitalize()} values for {network} network"
        else:
            for roi_data in rois.values():
                activation_data = roi_data['activations'].get(window)
                distance_data = roi_data['distances'].get(window)

                if activation_data is None or distance_data is None:
                    activation_values.append(None)
                    distance_values.append(None)
                    activation_std_devs.append(None)
                    distance_std_devs.append(None)
                else:
                    activation_values.append(activation_data['avg'])
                    distance_values.append(distance_data['avg'])
                    activation_std_devs.append(activation_data['std'])
                    distance_std_devs.append(distance_data['std'])

            title = f"Value for {network} network at window '{window}'"

        x = np.arange(len(roi_labels))
        width = 0.35

        fig, ax = plt.subplots(figsize=(40, 20))
        rects1 = ax.bar(x - width / 2, activation_values, width, label='Activations', color='skyblue',
                        yerr=activation_std_devs, capsize=5)
        rects2 = ax.bar(x + width / 2, distance_values, width, label='Distances', color='lightcoral',
                        yerr=distance_std_devs, capsize=5)

        ax.set_xlabel('ROIs', fontsize=30)
        ax.set_ylabel('Values', fontsize=30)
        ax.set_title(title, fontsize=30)
        ax.set_xticks(x - 0.7)
        ax.set_xticklabels(roi_labels, rotation=75, fontsize=30)
        ax.set_ylim(0, 1)
        ax.legend(fontsize=20)

        ax.tick_params(axis='y', labelsize=20)

        fig.tight_layout()
        # plt.savefig(f"figures/combined_resting_state/{title}.png")
        plt.show()


def plot_combined_roi_value_separate(activation_results, distance_results, method=None, window=None):
    if method not in ['mean', 'max'] and window is None:
        print("Invalid input. Please choose either 'mean' or 'max' as method or specify a window.")
        return

    roi_to_net_map = StaticData.ROI_TO_NETWORK_MAPPING
    networks = list(set(roi_to_net_map.values()))

    def aggregate_results(results):
        aggregated_values = {network: [] for network in networks}
        aggregated_std_devs = {network: [] for network in networks}
        aggregated_labels = []

        for roi, network in roi_to_net_map.items():
            roi_data = results.get(roi, {})
            if window is None:
                if method == 'mean':
                    avg_values = [roi_data[win]['avg'] for win in roi_data]
                    aggregated_values[network].append(sum(avg_values) / len(avg_values))
                    aggregated_std_devs[network].append(sum(roi_data[win]['std'] for win in roi_data) / len(roi_data))
                else:
                    max_values = [roi_data[win]['avg'] for win in roi_data]
                    aggregated_values[network].append(max(max_values))
                    aggregated_std_devs[network].append(max(roi_data[win]['std'] for win in roi_data))
            else:
                value_at_window = roi_data.get(window)
                if value_at_window is None:
                    aggregated_values[network].append(None)
                    aggregated_std_devs[network].append(None)
                else:
                    aggregated_values[network].append(value_at_window['avg'])
                    aggregated_std_devs[network].append(value_at_window['std'])
            aggregated_labels.append(roi)

        return aggregated_values, aggregated_std_devs, aggregated_labels

    activation_values, activation_std_devs, roi_labels = aggregate_results(activation_results)
    distance_values, distance_std_devs, _ = aggregate_results(distance_results)

    def plot_values(values, std_devs, title, filename):
        x = np.arange(len(roi_labels))
        width = 0.35

        fig, ax = plt.subplots(figsize=(40, 20))
        bottom = np.zeros(len(roi_labels))
        colors = plt.cm.get_cmap('tab20', len(networks))

        for i, network in enumerate(networks):
            network_values = values[network]
            network_std_devs = std_devs[network]
            bar = ax.bar(x, network_values, width, label=network, color=colors(i), yerr=network_std_devs, capsize=5,
                         bottom=bottom)
            bottom += np.nan_to_num(network_values)

        ax.set_xlabel('ROIs', fontsize=30)
        ax.set_ylabel('Values', fontsize=30)
        ax.set_title(title, fontsize=30)
        ax.set_xticks(x)
        ax.set_xticklabels(roi_labels, rotation=75, fontsize=10)
        ax.legend(fontsize=20)

        ax.tick_params(axis='y', labelsize=20)
        ax.set_ylim(0, 1)

        fig.tight_layout()
        # plt.savefig(f"figures/distances/{filename}.png")
        plt.show()

    plot_values(activation_values, activation_std_devs, 'Activation Values by Network', 'activation_values')
    plot_values(distance_values, distance_std_devs, 'Distance Values by Network', 'distance_values')


if __name__ == '__main__':
    # activation_res = pd.read_pickle("all_rois_groups_activations_results.pkl")
    # distance_res = pd.read_pickle("all_rois_groups_distances_results.pkl")
    # plot_combined_roi_value_separate(activation_res, distance_res, method="mean")
    # plot_combined_roi_value_separate(activation_res, distance_res, method="max")
    # plot_combined_roi_value_separate(activation_res, distance_res, window="13-18")

    activation_res = pd.read_pickle("all_rois_17_groups_10_sub_in_group_rest_between_data_5TR_window_activations_results.pkl")
    # distance_res = pd.read_pickle("all_rois_groups_distances_results_resting_state.pkl")
    plot_sorted_activations_single_plot(StaticData.ROI_TO_NETWORK_MAPPING, activation_res, window="13-18")
    plot_sorted_activations_single_plot(StaticData.ROI_TO_NETWORK_MAPPING, activation_res, window="0-5")
    activation_res = pd.read_pickle("all_rois_17_groups_10_sub_in_group_movie_data_last_5TR_activations_results.pkl")
    plot_sorted_activations_single_plot(StaticData.ROI_TO_NETWORK_MAPPING, activation_res, window="last_tr_movie")


#     # res = pd.read_pickle("all_rois_groups_activations_results.pkl")
#     # plot_roi_value(res, method="mean")
#     # plot_roi_value(res, method="max")
#     # plot_roi_value(res, window="13-18")
#
#     res = pd.read_pickle("all_rois_groups_distances_results.pkl")
#     plot_roi_value(res, window="13-18")
#     plot_roi_value(res, method="mean")
#     plot_roi_value(res, method="max")
