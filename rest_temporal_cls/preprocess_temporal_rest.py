import numpy as np
import pandas as pd
from tqdm import tqdm

from rest_temporal_cls.sequence_normalizer import get_normalized_data
from rest_temporal_cls.utils import generate_windows_pair


def _get_mean_subjects_group(data, **kwargs):
    # Convert subject IDs to a numpy array
    subject_ids_array = np.array(kwargs.get('subject_list'))

    # Use numpy.array_split to split the array into k groups
    k_subs = kwargs.get('k_subjects_in_group')
    subjects_group = np.array_split(subject_ids_array, k_subs)

    groups_data = {}
    for group_i, group in enumerate(subjects_group):
        preprocess_data = {}
        for rest_i in range(1, 15):
            rest_seq = data[
                (data['y'] == rest_i) & (data['timepoint'].isin(range(0, 19))) & (data['Subject'].isin(group))
                & (data['is_rest'] == 1)].copy()

            rest_seq = rest_seq.drop(['y', 'Subject', 'is_rest'], axis=1)
            rest_seq_mean_array = []
            for timepoint in range(0, 19):
                rest_seq_tr = rest_seq[rest_seq['timepoint'] == timepoint]
                rest_seq_tr = rest_seq_tr.drop('timepoint', axis=1)
                rest_seq_tr_mean = rest_seq_tr.values.mean(axis=0)
                rest_seq_mean_array.append(rest_seq_tr_mean)

            rest_seq_mean_df = pd.DataFrame(np.stack(rest_seq_mean_array))
            rest_seq_mean_df['timepoint'] = [*range(0, 19)]

            preprocess_movie = {}
            for w_i, (window_start, window_end) in enumerate(generate_windows_pair(k=5, n=18)):
                window_seq = rest_seq_mean_df[rest_seq_mean_df['timepoint'].isin(range(window_start, window_end))]
                window_seq = window_seq.copy().drop('timepoint', axis=1)
                preprocess_movie[f'{window_start}-{window_end}'] = window_seq.values.flatten()

            preprocess_data[rest_i] = preprocess_movie

        groups_data[group_i] = preprocess_data

    return groups_data


def get_temporal_rest_window_activations(roi: str, **kwargs):
    normalized_data = get_normalized_data(roi=roi, **kwargs)
    subjects = normalized_data['Subject'].unique()
    print("Preprocessing Data")

    if kwargs.get('group_average'):
        return _get_mean_subjects_group(normalized_data, subject_list=subjects, **kwargs)

    subjects_data = {}
    for sub in tqdm(subjects):
        preprocess_data = {}
        for rest_i in range(1, 15):
            rest_seq = normalized_data[
                (normalized_data['y'] == rest_i)
                & (normalized_data['timepoint'].isin(range(0, 19)))
                & (normalized_data['Subject'] == sub)
                & (normalized_data['is_rest'] == 1)
                ]

            preprocess_movie = {}
            for w_i, (window_start, window_end) in enumerate(generate_windows_pair(k=5, n=18)):
                window_seq = rest_seq[rest_seq['timepoint'].isin(range(window_start, window_end))]
                window_seq = window_seq.drop(['y', 'timepoint', 'Subject', 'is_rest'], axis=1)
                preprocess_movie[f'{window_start}-{window_end}'] = window_seq.values.flatten()

            preprocess_data[rest_i] = preprocess_movie

        subjects_data[sub] = preprocess_data

    return subjects_data
