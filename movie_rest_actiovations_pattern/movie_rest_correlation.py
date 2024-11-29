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
from io import BytesIO

import boto3
import matplotlib.pyplot as plt
import numpy as np
# Main function to execute the entire algorithm
import pandas as pd
import tqdm

from config import BUCKET_NAME
from dataloader.sequence_normalizer import get_normalized_data
from static_data.static_data import StaticData
import aws_utils.upload_s3 as s3_utils


def get_group_average(group_list, data, feat_columns):
    subjects_data = []
    timepoints = []
    movies = []
    for sub in group_list:
        subject_data = data[data['Subject'] == sub]
        subject_data_feat = subject_data[feat_columns]
        subjects_data.append(subject_data_feat)
        timepoints = subject_data["timepoint"].values
        movies = subject_data["y"].values

    group_mean = pd.DataFrame(np.mean(subjects_data, axis=0))
    group_mean.columns = feat_columns
    group_mean["timepoint"] = timepoints
    group_mean["y"] = movies

    return group_mean


def get_activations(data, n_groups=17, subjects_per_group=10):
    # Step 1: Identify feature columns
    non_features_column = ['y', 'timepoint', 'Subject', 'is_rest']
    feature_columns = [col for col in data.columns if col not in non_features_column]

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
        movie_data_mean = get_group_average(group_list=group_subjects, data=movie_data, feat_columns=feature_columns)

        # Step 6: Average features for the last 5 timepoints for each movie
        movie_vectors = []  # Initialize a list for movie average feature vectors
        grouped_movies = movie_data_mean.groupby('y')
        for movie_id, movie_group in grouped_movies:
            last_five_timepoints = movie_group.tail(5)[feature_columns]
            average_features = last_five_timepoints.mean()
            movie_vectors.append(average_features)

        # Convert to DataFrame and flatten
        movie_vectors_df = pd.DataFrame(movie_vectors).fillna(0)
        concatenated_movies = movie_vectors_df.values.flatten()

        # Step 8: Process rest data (is_rest == 1) with sliding window of 5 timepoints
        rest_data = group[group['is_rest'] == 1]
        rest_data_mean = get_group_average(group_list=group_subjects, data=rest_data, feat_columns=feature_columns)

        # Create 14 rest windows, averaging per subject for each window
        concatenated_rest_windows = []
        for i in range(14):
            # Select window data based on the current timepoint range
            window_data = rest_data_mean[rest_data_mean["timepoint"].isin(range(i, i + 5))]
            # Average features per movie, per subject in this window
            rest_averaged_features = []  # List to hold averaged features per movie
            for movie in window_data['y'].unique():
                rest_group_window = window_data[window_data['y'] == movie][feature_columns]
                rest_group_window_avg = rest_group_window.mean()
                rest_averaged_features.append(rest_group_window_avg)

            concat_rest_windows = pd.DataFrame(rest_averaged_features).fillna(0)
            concat_rest_windows = concat_rest_windows.values.flatten()
            concatenated_rest_windows.append(concat_rest_windows)

        # Step 9: Compute correlations between concatenated movie vector and each concatenated rest window
        correlations = []
        for rest_window in concatenated_rest_windows:
            # todo: make sure its pearson. debug RH_vis13 - LAST WINDOW -0.4
            corr = np.corrcoef(concatenated_movies, rest_window)[0, 1]  # Correlate
            correlations.append(corr)

        # Step 10: Store the correlation vector for this group
        all_correlations.append(correlations)

        # Step 11: Compute the mean correlation across all groups
    mean_correlation_vector = np.mean(all_correlations, axis=0)
    std_correlation_vector = np.std(all_correlations, axis=0, ddof=1) / (np.sqrt(n_groups) / 2)

    # _plot(mean_correlation_vector, std_correlation_vector)

    return all_correlations, mean_correlation_vector, std_correlation_vector


def _plot(mean_vector, std_vector):
    # Generate x-axis values
    x = np.arange(len(mean_vector))

    # Create the plot
    plt.figure(figsize=(8, 5))
    plt.errorbar(x, mean_vector, yerr=std_vector, fmt='-o', capsize=5, elinewidth=2,
                 markeredgewidth=2)

    # Add labels and title
    plt.xlabel("X-Axis Label (e.g., Groups or Timepoints)")
    plt.ylabel("Mean Correlation")
    plt.title("Mean Correlation with Error Bars")
    plt.grid(alpha=0.5)
    plt.tight_layout()

    # Show the plot
    plt.show()


def main():
    # Upload the pickle file to S3
    s3 = boto3.client('s3')
    bucket_name = 'erezsimony'
    StaticData.inhabit_class_members()
    roi_list = StaticData.ROI_NAMES
    for roi in tqdm.tqdm(roi_list):
        s3_destination_path = f"Results/negative_correlation/{roi}.pkl"
        if s3_utils.file_exists(bucket_name=BUCKET_NAME, s3_path=s3_destination_path):
            continue

        n_groups = 17
        roi_norm = get_normalized_data(roi)
        window_corr, mean_activations_vector, std_activation_vector = get_activations(
            data=roi_norm, subjects_per_group=10, n_groups=n_groups
        )
        # Combine the results into a dictionary
        results = {
            'mean': mean_activations_vector,
            'std': std_activation_vector,
            'activations_vector': window_corr
        }

        # Save the results to a pickle file
        with open(f'negative_correlation/{roi}.pkl', 'wb') as f:
            pickle.dump(results, f)

        file_name = f'/Users/nivyahav/projects/fmri_resting_state_activation_classification/negative_correlation/{roi}.pkl'
        try:
            s3.upload_file(file_name, bucket_name, s3_destination_path)
            print(f"File {file_name} uploaded successfully to {s3_destination_path}")
        except Exception as e:
            print(f"Failed to upload {file_name} to S3: {e}")

def upload_csv_results():
    """
    Reads pickle files for each ROI from S3, extracts the mean value from the last window,
    and uploads a single CSV file with all ROIs' last window mean results to S3.
    """
    s3 = boto3.client('s3')
    bucket_name = BUCKET_NAME
    output_csv_path = "Results/last_window_mean_results.csv"

    StaticData.inhabit_class_members()
    roi_list = StaticData.ROI_NAMES
    # Initialize a list to collect all ROI results
    last_window_results = []

    for roi in tqdm.tqdm(roi_list):
        s3_source_path = f"Results/negative_correlation/{roi}.pkl"

        # Check if the pickle file exists in S3
        if not s3_utils.file_exists(bucket_name=bucket_name, s3_path=s3_source_path):
            print(f"File not found in S3: {s3_source_path}")
            continue

        # Download the pickle file
        response = s3.get_object(Bucket=bucket_name, Key=s3_source_path)
        results = pickle.load(BytesIO(response['Body'].read()))

        # Extract the mean vector
        if "activations_vector" not in results:
            print(f"No mean vector found in {s3_source_path}")
            continue

        # Get the last value of the mean vector
        last_window_mean = results['mean'][-1]
        last_window_results.append({"ROI": roi, "value": last_window_mean})

    # Convert the results into a DataFrame
    df_results = pd.DataFrame(last_window_results).sort_values(by=['value'], ascending=False).reset_index(drop=True)

    # Save the DataFrame to a local CSV file
    local_csv_path = "last_window_mean_results.csv"
    df_results.to_csv(local_csv_path, index=False)

    # Upload the CSV to S3
    try:
        s3.upload_file(local_csv_path, bucket_name, output_csv_path)
        print(f"CSV uploaded successfully to s3://{bucket_name}/{output_csv_path}")
    except Exception as e:
        print(f"Failed to upload CSV to S3: {e}")


if __name__ == '__main__':
    main()
    upload_csv_results()
