import os.path

import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.svm import SVC

from movie_temporal_cls.process_temporal_movie import get_temporal_movie_window_activations
from utils import Utils

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


def evaluate_movie_windows(
        rois: List[str],
        distances: bool,
        checkpoint: bool,
        validation: Literal["k_fold", "llo"],
        group_average: bool,
        **kwargs
):
    """
    Evaluates the performance of a machine learning model on temporal movie windows.

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
    window_name = "last_movie_window"
    for roi in tqdm(rois):
        # Skip processed ROI if checkpointing is enabled and data is loaded from the file.
        if loaded_data.get(roi) and checkpoint:
            rois_results[roi] = loaded_data.pop(roi)
            continue

        print(f"Training ROI: {roi}")
        window_score = {}
        data = get_temporal_movie_window_activations(roi=roi, group_average=group_average, **kwargs)
        total_scores = []
        if group_average or kwargs.get('group_mean_correlation'):
            split_group = [*range(len(data))]
        else:
            split_group = subjects_group

        for train_index, test_index in tqdm(validation_split.split(split_group)):
            train_group = [split_group[i] for i in train_index]
            test_group = [split_group[i] for i in test_index]

            train_data = create_subject_group_dataset(data, window=window_name, subjects_group=train_group,
                                                      distances=distances)
            test_data = create_subject_group_dataset(data, window=window_name, subjects_group=test_group,
                                                     distances=distances)
            model = train_window_k(dataset_df=train_data, shuffle=False)
            score = evaluate_window_k(model=model, dataset_df=test_data, shuffle=False)
            total_scores.append(score)

        np_scores = np.array(total_scores)
        avg_score = np.mean(np_scores)
        std_score = np.std(np_scores, ddof=1) / np.sqrt(len(subjects_group))
        rois_results[roi] = {'avg': avg_score, 'std': std_score}

        if checkpoint:
            with open(file_output_name, 'wb') as file:
                pickle.dump(rois_results, file)

    # Utils.plot_roi_temporal_windows_dynamic(rois_results, mode='distances' if distances else 'activations')


def create_subject_group_dataset(data, window, subjects_group: List, distances: bool):
    dataset_df = pd.DataFrame()

    for subject in subjects_group:
        subject_data = data.get(int(subject))
        subject_df = pd.DataFrame()
        for movie_i in subject_data:
            movie_features = subject_data.get(movie_i).get(window)
            movie_df = pd.DataFrame(movie_features).transpose()
            movie_df['y'] = movie_i
            subject_df = pd.concat([subject_df, movie_df])

        if distances:
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

    model = SVC()
    model.fit(x_train, y_train)

    return model


def evaluate_window_k(model: SVC, dataset_df: pd.DataFrame, shuffle: bool):
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
