import pickle
from typing import Tuple, List

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.svm import LinearSVC
from tqdm import tqdm

from rest_temporal_cls.preprocess_temporal_rest import get_temporal_rest_window_activations
from rest_temporal_cls.utils import generate_windows_pair, Utils

accumulated_scores = {}

from sklearn.model_selection import LeaveOneOut


def create_subject_group_dataset(data, window_range: Tuple[int, int], subjects_group: List):
    w_s, w_e = window_range
    dataset_df = pd.DataFrame()

    for subject in subjects_group:
        subject_data = data.get(int(subject))
        for rest_i in subject_data:
            movie_features = subject_data.get(rest_i).get(f'{w_s}-{w_e}')
            movie_df = pd.DataFrame(movie_features).transpose()
            movie_df['y'] = rest_i
            dataset_df = pd.concat([dataset_df, movie_df])
    return dataset_df


def plot_window_score(roi, window_score):
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

def get_windows_distances_data():
    pass
def get_windows_activations_data():
    rois = ['RH_DorsAttn_Post_2', 'RH_Default_pCunPCC_1', 'RH_Vis_18']
    subjects_group = Utils.subject_list

    loo = LeaveOneOut()

    rois_results = {}
    for roi in rois:
        window_score = {}
        data = get_temporal_rest_window_activations(roi=roi)
        for window_s, window_e in generate_windows_pair(k=5, n=19):
            total_scores = []
            window_as_str = f'{window_s}-{window_e}'
            print(f"Training window: {window_as_str}")
            for train_index, test_index in tqdm(loo.split(subjects_group)):
                train_group = [subjects_group[i] for i in train_index]
                test_group = [subjects_group[test_index[0]]]  # Single test subject

                train_data = create_subject_group_dataset(data, window_range=(window_s, window_e),
                                                          subjects_group=train_group)
                test_data = create_subject_group_dataset(data, window_range=(window_s, window_e),
                                                         subjects_group=test_group)
                model = train_window_k(train_data)
                score = evaluate_window_k(model, test_data, (window_s, window_e))
                total_scores.append(score)

            np_scores = np.array(total_scores)
            avg_score = np.mean(np_scores)
            std_score = np.std(np_scores, ddof=1) / np.sqrt(len(subjects_group))
            window_score[window_as_str] = {}
            window_score[window_as_str]['avg'] = avg_score
            window_score[window_as_str]['std'] = std_score
            print(f'Overall Average Score for window: {window_as_str}', avg_score)

        rois_results[roi] = window_score
        # plot_window_score(roi, window_score)
        # break  # Remove this line if you want to iterate through all ROIs

    with open('rois_activations_results.pkl', 'wb') as file:
        pickle.dump(rois_results, file)


def train_window_k(dataset_df):
    dataset_df = dataset_df.sample(frac=1).reset_index(drop=True)
    y_train = dataset_df['y'].values
    x_train = dataset_df.drop(['y'], axis=1).values

    model = LinearSVC()
    model.fit(x_train, y_train)

    return model


def evaluate_window_k(model, test_data, window_indices):
    dataset_df = test_data.sample(frac=1).reset_index(drop=True)
    y_test = dataset_df['y'].values
    x_test = dataset_df.drop(['y'], axis=1).values
    score = model.score(x_test, y_test)
    return round(score, 3)


if __name__ == '__main__':
    get_windows_activations_data()
