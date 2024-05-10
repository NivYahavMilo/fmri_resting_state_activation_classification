import matplotlib.pyplot as plt
import pandas as pd


def plot_roi_value(results, method=None, window=None):
    if method not in ['mean', 'max'] and window is None:
        print("Invalid input. Please choose either 'mean' or 'max' as method or specify a window.")
        return

    roi_labels = list(results.keys())
    values = []
    std_devs = []

    if window is None:
        for roi in results.values():
            if method == 'mean':
                avg_values = [roi[win]['avg'] for win in roi]
                values.append(sum(avg_values) / len(avg_values))
                std_devs.append(sum(roi[win]['std'] for win in roi) / len(roi))
            else:
                max_values = [roi[win]['avg'] for win in roi]
                values.append(max(max_values))
                std_devs.append(max(roi[win]['std'] for win in roi))

        title = f"{method.capitalize()} values for ROIs"
    else:
        for roi in results.values():
            value_at_window = roi.get(window)
            if value_at_window is None:
                values.append(None)
                std_devs.append(None)
            else:
                values.append(value_at_window['avg'])
                std_devs.append(value_at_window['std'])
        title = f"Value for ROIs at window '{window}'"

    plt.figure(figsize=(12, 8))
    plt.bar(roi_labels, values, color='skyblue', yerr=std_devs, capsize=5)
    plt.title(title)
    plt.xlabel('ROIs')
    plt.ylabel('Value')
    plt.xticks(rotation=45)
    plt.show()


if __name__ == '__main__':
    res = pd.read_pickle("rois_activations_results_avg_3.pkl")
    plot_roi_value(res, method="mean")
