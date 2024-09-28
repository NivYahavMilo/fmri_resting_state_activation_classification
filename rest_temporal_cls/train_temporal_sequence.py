import os.path
from typing import Tuple

import pandas as pd
import scipy
from matplotlib import pyplot as plt
from sklearn.exceptions import ConvergenceWarning
from sklearn.svm import LinearSVC

from rest_temporal_cls.preprocess_temporal_rest import get_temporal_rest_window_activations
from utils import generate_windows_pair, Utils

accumulated_scores = {}
import warnings

# Suppress convergence warnings
warnings.filterwarnings("ignore", category=ConvergenceWarning)

import os
import pickle
from typing import List, Literal
from tqdm import tqdm
import numpy as np
from sklearn.model_selection import KFold, LeaveOneOut

def evaluate_rest_windows(
        rois: List[str],
        distances: bool,
        checkpoint: bool,
        validation: Literal["k_fold", "llo"],
        group_average: bool,
        **kwargs
):
    """
    Evaluates the performance of a machine learning model on temporal rest windows.

    Parameters:
    - rois (List): List of Regions of Interest (ROIs) to evaluate.
    - distances (bool): True if evaluating distances, False for activations.
    - checkpoint (bool): Whether to load saved results from a file.
    - validation (Literal): Type of validation ("k_fold" or "llo").
    - group_average (bool): Whether to use group averaging.
    - **kwargs: Additional parameters.

    Returns:
    - None
    """
    file_output_name = kwargs.get('output_name')
    loaded_data = {}
    if checkpoint and os.path.isfile(file_output_name):
        with open(file_output_name, 'rb') as output_file_io:
            loaded_data = pickle.load(output_file_io)

    if validation == 'k_fold':
        validation_k_split = kwargs.get('k_split', 8)
        validation_split = KFold(n_splits=validation_k_split, shuffle=True, random_state=42)
    elif validation == 'llo':
        validation_split = LeaveOneOut()
    else:
        raise NotImplementedError(f"Validation method must be one of the following: {validation}")

    rois_results = {}
    subjects_group = Utils.subject_list
    k_window_size = kwargs.get('k_window_size')
    n_timepoints = kwargs.get('n_timepoints')
    for roi in tqdm(rois):
        # Skip processed ROI if checkpointing is enabled and data is loaded from the file.
        if loaded_data.get(roi) and checkpoint:
            rois_results[roi] = loaded_data.pop(roi)
            continue

        print(f"Training ROI: {roi}")
        window_score = {}
        data = get_temporal_rest_window_activations(roi=roi, group_average=group_average, **kwargs)
        for window_s, window_e in generate_windows_pair(k=k_window_size, n=n_timepoints):
            total_scores = []
            window_as_str = f'{window_s}-{window_e}'
            print(f"Training window: {window_as_str}")
            if group_average or kwargs.get('group_mean_correlation'):
                split_group = [*range(len(data))]
            else:
                split_group = subjects_group

            for train_index, test_index in tqdm(validation_split.split(split_group)):
                train_group = [split_group[i] for i in train_index]
                test_group = [split_group[i] for i in test_index]

                train_data = create_subject_group_dataset(
                    data, window_range=(window_s, window_e), subjects_group=train_group, distances=distances,
                    correlation_mean=kwargs.get('group_mean_correlation')
                )
                test_data = create_subject_group_dataset(
                    data, window_range=(window_s, window_e), subjects_group=test_group, distances=distances,
                    correlation_mean=kwargs.get('group_mean_correlation')
                )
                model = train_window_k(dataset_df=train_data, shuffle=False)
                score = evaluate_window_k(model=model, dataset_df=test_data, shuffle=False)
                total_scores.append(score)

            np_scores = np.array(total_scores)
            avg_score = np.mean(np_scores)
            std_score = np.std(np_scores, ddof=1) / np.sqrt(len(subjects_group))
            window_score[window_as_str] = {'avg': avg_score, 'std': std_score}
            print(f'Overall Average Score for window: {window_as_str}', avg_score)

        rois_results[roi] = window_score

        if checkpoint:
            with open(file_output_name, 'wb') as file:
                pickle.dump(rois_results, file)

    # Utils.plot_roi_temporal_windows_dynamic(rois_results, mode='distances' if distances else 'activations')


def create_subject_group_dataset(data, window_range: Tuple[int, int], subjects_group: List, distances: bool,
                                 correlation_mean: bool):
    """
    Creates a dataset for machine learning from temporal rest window activations.

    Parameters:
    - data: Data containing temporal rest window activations.
    - window_range (Tuple): Tuple specifying the start and end of the window.
    - subjects_group (List): List of subjects for the dataset.
    - distances (bool): True if creating a dataset for distances, False for activations.

    Returns:
    - Pandas DataFrame representing the dataset.
    """
    w_s, w_e = window_range
    dataset_df = pd.DataFrame()

    for subject in subjects_group:
        subject_data = data.get(int(subject))
        subject_df = pd.DataFrame()
        if correlation_mean:
            dist_seq = subject_data[f'{w_s}-{w_e}']
            dist_seq_df = pd.DataFrame(dist_seq)
            dist_seq_df['y'] = [*range(1, 15)]
            subject_df = dist_seq_df

        else:
            for rest_i in subject_data:
                movie_features = subject_data.get(rest_i).get(f'{w_s}-{w_e}')
                movie_df = pd.DataFrame(movie_features).transpose()
                movie_df['y'] = rest_i
                subject_df = pd.concat([subject_df, movie_df])

        if distances and not correlation_mean:
            subject_df_copy = subject_df.copy()
            subject_df_copy = subject_df_copy.drop('y', axis=1)
            subject_df_copy = subject_df_copy.transpose()
            subject_correlation = subject_df_copy.corr()
            subject_correlation = pd.DataFrame(np.array([row[row < 1] for row in subject_correlation.values]))
            subject_correlation['y'] = subject_df['y'].to_list()
            subject_df = subject_correlation

        dataset_df = pd.concat([dataset_df, subject_df])
    return dataset_df


def train_window_k(dataset_df: pd.DataFrame, shuffle: bool):
    """
    Trains a machine learning model on a given dataset.

    Parameters:
    - dataset_df: Pandas DataFrame representing the dataset.

    Returns:
    - Trained machine learning model.
    """
    if shuffle:
        dataset_df = dataset_df.sample(frac=1).reset_index(drop=True)

    y_train = dataset_df['y'].values
    x_train = dataset_df.drop(['y'], axis=1).values

    model = LinearSVC()
    model.fit(x_train, y_train)

    return model


def evaluate_window_k(model: LinearSVC, dataset_df: pd.DataFrame, shuffle: bool):
    """
    Evaluates the performance of a machine learning model on a test dataset.

    Parameters:
    - model: Trained machine learning model.
    - test_data: Pandas DataFrame representing the test dataset.

    Returns:
    - Accuracy score on the test dataset.
    """
    if shuffle:
        dataset_df = dataset_df.sample(frac=1).reset_index(drop=True)

    y_test = dataset_df['y'].values
    x_test = dataset_df.drop(['y'], axis=1).values
    score = model.score(x_test, y_test)
    return round(score, 3)


def plot_window_score(roi, window_score):
    """
     Plots a bar chart showing the scores for different windows.

     Parameters:
     - roi (str): Region of Interest (ROI) label.
     - window_score (dict): Dictionary containing window-wise scores.

     Returns:
     - None
     """
    # Extract keys and values from the dictionary
    keys = list(window_score.keys())
    values = list(window_score.values())

    # Create a bar plot
    plt.figure(figsize=(8, 6))
    plt.plot(keys, values)
    plt.xlabel('windows')
    plt.ylabel('scores')
    plt.title(f'Roi: {roi} - Accuracy as function of window')
    plt.show()


def train_k_fold_mat_file():
    # Load the .mat file
    mat_data = scipy.io.loadmat('rest_dist_all_win_new_all.mat')
    labels = scipy.io.loadmat('task_label.mat')

    # Extract the 3D tensor
    x = mat_data['rest_dist_all_win_new_all']
    y = labels['task_label']

    llo = LeaveOneOut()
    window_score = {}
    for window_s, window_e in generate_windows_pair(k=5, n=18):
        window_as_str = f'{window_s}-{window_e}'
        print(f"Training window: {window_as_str}")

        x_window = x[window_s, :, :]
        df = pd.DataFrame(x_window)
        df['y'] = y
        # Initialize Group column
        df['group'] = 0

        # Identify groups based on sequence of intervals
        group_num = 0
        prev_interval = None

        for i, interval in enumerate(df['y']):
            if interval == 1 and prev_interval == 14:
                group_num += 1
            df.at[i, 'group'] = group_num

            prev_interval = interval

        groups = range(group_num)
        total_scores = []

        for train_index, test_index in tqdm(llo.split(groups)):
            train_group = [groups[i] for i in train_index]
            test_group = [groups[i] for i in test_index]

            df_train = df[df['group'].isin(train_group)]
            df_test = df[df['group'].isin(test_group)]
            model = train_window_k(dataset_df=df_train, shuffle=False)
            score = evaluate_window_k(model=model, dataset_df=df_test, shuffle=False)
            total_scores.append(score)

        np_scores = np.array(total_scores)
        avg_score = np.mean(np_scores)
        std_score = np.std(np_scores, ddof=1) / np.sqrt(len(groups))
        window_score[window_as_str] = {}
        window_score[window_as_str]['avg'] = avg_score
        window_score[window_as_str]['std'] = std_score
        print(f'Overall Average Score for window: {window_as_str}', avg_score)

    #   Utils.plot_roi_temporal_windows_dynamic(window_score, mode='distances', roi='Dorsal Attention')

