# normalized data per scan
# devied to group - 17 groups 10 subject
# average each group - get 17 "subjects"
# group 1  -
# 5 movie last tr - average 5 tr - we get a single avg vector for each movie
# concat  the movies avg vector (single vector per ROI)
# take each rest window and repeat the process.
# perform correlation between (movie vector| rest vector(i,j)) window.
# 14 sized vector
# perform mean on 14 sized vector accros all groups.
# plot line with the correlation values accross windows.
import pickle

import boto3
import numpy as np
# Main function to execute the entire algorithm
import pandas as pd
import tqdm

from dataloader.sequence_normalizer import get_normalized_data
from static_data.static_data import StaticData


def get_activations(data, n_groups=17, subjects_per_group=10):
    # Step 1: Identify feature columns
    feature_columns = [col for col in data.columns if col not in ['y', 'timepoint', 'Subject', 'is_rest']]

    # Step 2: Split dataset into groups of subjects
    unique_subjects = data['Subject'].unique()
    all_correlations = []

    for group_idx in range(n_groups):
        # Step 3: Select the subjects for the current group
        group_subjects = unique_subjects[group_idx * subjects_per_group:(group_idx + 1) * subjects_per_group]

        # Step 4: Filter the data to get only the current group of subjects
        group = data[data['Subject'].isin(group_subjects)]

        # Step 5: Process movie data (is_rest == 0)
        movie_data = group[group['is_rest'] == 0]

        # Step 6: Average features for the last 5 timepoints for each movie
        movie_vectors = []  # Initialize a list for movie average feature vectors

        grouped_movies = movie_data.groupby('y')
        for movie_id, movie_group in grouped_movies:
            last_five_timepoints = movie_group.tail(5)[feature_columns]
            average_features = last_five_timepoints.mean()
            movie_vectors.append(average_features)

        # Convert to DataFrame and flatten
        movie_vectors_df = pd.DataFrame(movie_vectors)
        concatenated_movies = movie_vectors_df.values.flatten()

        # Step 8: Process rest data (is_rest == 1) with sliding window of 5 timepoints
        rest_data = group[group['is_rest'] == 1]
        rest_windows = []

        # Create 14 rest windows, averaging per subject for each window
        concatenated_rest_windows = []
        for i in range(14):
            # Select window data based on the current timepoint range
            window_data = rest_data[rest_data["timepoint"].isin(range(i, i + 5))]

            # Initialize a list for averaged features for this window
            averaged_window = []

            # Average features per movie, per subject in this window
            movie_averaged_features = []  # List to hold averaged features per movie
            for movie in window_data['y'].unique():
                movie_data = window_data[window_data['y'] == movie]
                movie_subject_means = []  # List to hold features for each subject for the current movie

                for subject_id in movie_data['Subject'].unique():
                    subject_features = movie_data[movie_data['Subject'] == subject_id][feature_columns]
                    movie_subject_mean = subject_features.mean()  # Average features for this subject
                    movie_subject_means.append(movie_subject_mean)

                movie_mean = pd.DataFrame(movie_subject_means).mean()
                movie_averaged_features.append(movie_mean)

            concatenated_rest_windows.append(pd.DataFrame(movie_averaged_features).values.flatten())

        # Step 9: Compute correlations between concatenated movie vector and each concatenated rest window
        correlations = []
        for rest_window in concatenated_rest_windows:
            corr = np.corrcoef(concatenated_movies, rest_window)[0, 1]  # Correlate
            correlations.append(corr)

        # Step 10: Store the correlation vector for this group
        all_correlations.append(correlations)

        # Step 11: Compute the mean correlation across all groups
    mean_correlation_vector = np.mean(all_correlations, axis=0)

    return mean_correlation_vector


def get_movie_rest_activations(roi: str):
    n_groups = 17
    roi_norm = get_normalized_data(roi)
    activations_vector = get_activations(data=roi_norm, subjects_per_group=10, n_groups=n_groups)
    # Compute mean and standard deviation, dividing std by sqrt of the number of groups
    mean_activations = np.mean(activations_vector)
    std_activations = np.std(activations_vector, ddof=1) / np.sqrt(n_groups)
    return activations_vector, mean_activations, std_activations


def main():
    # Upload the pickle file to S3
    s3 = boto3.client('s3')
    bucket_name = 'erezsimony'
    StaticData.inhabit_class_members()
    roi_list = StaticData.ROI_NAMES
    for roi in tqdm.tqdm(roi_list):
        activations_vec, mean_vec, std_vec = get_movie_rest_activations(roi=roi)

        # Combine the results into a dictionary
        results = {
            'mean': mean_vec,
            'std': std_vec,
            'activations_vector': activations_vec
        }

        # Save the results to a pickle file
        with open(f'negative_correlation/{roi}.pkl', 'wb') as f:
            pickle.dump(results, f)

        file_name = f'/Users/nivyahav/projects/fmri_resting_state_activation_classification/negative_correlation/{roi}.pkl'
        s3_destination_path = f"Results/negative_correlation/{roi}.pkl"
        try:
            s3.upload_file(file_name, bucket_name, s3_destination_path)
            print(f"File {file_name} uploaded successfully to {s3_destination_path}")
        except Exception as e:
            print(f"Failed to upload {file_name} to S3: {e}")


if __name__ == '__main__':
    main()
