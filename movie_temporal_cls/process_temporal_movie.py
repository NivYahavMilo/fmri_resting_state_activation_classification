import numpy as np
import pandas as pd

from dataloader.sequence_normalizer import get_normalized_data


def get_temporal_movie_window_activations(roi: str, **kwargs):
    normalized_data = get_normalized_data(roi=roi)
    subjects = normalized_data['Subject'].unique()
    print("Preprocessing Data...")
    return _get_single_last_window(data=normalized_data, subjects_list=subjects, **kwargs)


def _get_single_last_window(data, subjects_list, **kwargs):
    # Convert subject IDs to a numpy array
    subject_ids_array = np.array(subjects_list)

    # Use numpy.array_split to split the array into k groups
    k_subs = kwargs.get('k_split')
    subjects_group = np.array_split(subject_ids_array, k_subs)

    groups_data = {}
    for group_i, group in enumerate(subjects_group):
        preprocess_data = {}
        for movie_index in range(1, 15):
            movie_data = data[
                (data['y'] == movie_index) & (data['Subject'].isin(group))
                & (data['is_rest'] == 0)].copy()

            movie_data = movie_data.drop(['y', 'Subject', 'is_rest'], axis=1)
            movie_seq_mean_array = []
            for timepoint in movie_data.timepoint.unique():
                movie_seq_tr = movie_data[movie_data['timepoint'] == timepoint]
                movie_seq_tr = movie_seq_tr.drop('timepoint', axis=1)
                movie_seq_tr = movie_seq_tr.fillna(value=0)
                movie_seq_tr_mean = movie_seq_tr.values.mean(axis=0)
                movie_seq_mean_array.append(movie_seq_tr_mean)

            last_window_movie_mean_df = pd.DataFrame(np.stack(movie_seq_mean_array))
            last_window_movie_mean_df["timepoint"] = [*range(len(movie_data)//len(group))]
            last_window_movie = last_window_movie_mean_df.tail(kwargs['k_window_size'])

            preprocess_movie = {}
            window_seq = last_window_movie.copy().drop('timepoint', axis=1)
            if kwargs['window_preprocess_method'] == 'mean':
                preprocess_movie['last_movie_window'] = window_seq.mean(axis=0)
            elif kwargs['window_preprocess_method'] == 'flattening':
                preprocess_movie["last_movie_window"] = window_seq.values.flatten()

            preprocess_data[movie_index] = preprocess_movie
        groups_data[group_i] = preprocess_data

    return groups_data