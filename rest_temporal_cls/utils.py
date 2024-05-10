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
        # plt.ylim(0, 1)

        # Show the plot
        plt.show()


def generate_windows_pair(k: int, n: int):
    for i in range(n - k + 1):
        if i + k <= n:
            yield i, i + k


