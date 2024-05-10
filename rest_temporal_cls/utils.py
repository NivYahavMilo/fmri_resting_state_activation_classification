import os

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

import config
from static_data.static_data import StaticData


class Utils:
    StaticData().inhabit_class_members()

    roi_list = StaticData.ROI_NAMES
    subject_list = StaticData.SUBJECTS
    movie_scan_mapping = StaticData.SCAN_MAPPING

    @staticmethod
    def plot_roi_temporal_windows_dynamic(data: dict, mode: str, roi: str = None):
        if len(data.keys()) == 14:
            roi_data = data
            data = {roi: roi_data}

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
        #plt.ylim(0, 1)

        # Show the plot
        plt.show()


def generate_windows_pair(k: int, n: int):
    for i in range(n - k + 1):
        if i + k <= n:
            yield i, i + k


def plot_roi_temporal_windows_dynamic(results, **kwargs):
    roi_list = kwargs.get('roi_list', [])
    n_roi = len(roi_list)
    for roi in roi_list:
        roi_scores = results.get(roi)
        n_windows = len(roi_scores)
        for window, scores in roi_scores:
            y = np.zeros(shape=(n_roi, len(n_windows)))
        for group_index, dynamic_dict in results.items():
            y[:, group_index - 1] = [*dynamic_dict.values()]
            x_ticks = [*dynamic_dict.keys()]

    y = np.nan_to_num(y)
    mean_dynamic = y.mean(axis=1)
    std_dynamic = y.std(axis=1, ddof=1) / np.sqrt(y.shape[1])

    sns.set()
    sns.set_theme(style="darkgrid")
    fig = plt.gcf()
    plt.plot(range(0, len(mean_dynamic)), mean_dynamic, linewidth=4, color='blue')

    plt.fill_between(x=range(0, len(mean_dynamic)),
                     y1=np.array(mean_dynamic) + np.array(std_dynamic),
                     y2=np.array(mean_dynamic) - np.array(std_dynamic),
                     facecolor='blue',
                     alpha=0.2)

    title = f"Mean of 6 groups dynamic over 29-30 average subjects\n ROI={kwargs.get('roi')}\nCLIP_WINDOW={kwargs.get('init_window').replace('dynamic', '')}TR"
    plt.title(title)
    plt.xticks(np.arange(len(x_ticks)), x_ticks, rotation=45)
    plt.ylim([-.4, .9])
    plt.xlabel('Rest TR window')
    plt.ylabel('Correlation Value')
    plt.legend([kwargs.get('analysis_mode').value, 'shuffle'])

    plt.draw()
    plt.show()
    if not os.path.isfile(config.ROOT_PATH):
        fig.savefig('image.png', dpi=300)
