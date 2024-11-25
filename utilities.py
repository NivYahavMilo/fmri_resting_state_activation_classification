import io
import os
import pickle

import boto3
import pandas as pd
import scipy.io
import tqdm

import config
from enums import Mode
from static_data.static_data import StaticData

StaticData.inhabit_class_members()


def get_clip_index_mapping(inverse: bool = False):
    clip_mapper = {
        "0": "testretest",
        "1": "twomen",
        "2": "bridgeville",
        "3": "pockets",
        "4": "overcome",
        "5": "inception",
        "6": "socialnet",
        "7": "oceans",
        "8": "flower",
        "9": "hotel",
        "10": "garden",
        "11": "dreary",
        "12": "homealone",
        "13": "brokovich",
        "14": "starwars"
    }

    if inverse:
        clip_mapper = {v: k for k, v in clip_mapper.items()}

    return clip_mapper


def convert_pkl_to_mat(mode: Mode, roi: list[str]):
    raw_data_mat = {}
    raw_data_path = config.SUBNET_DATA_DF.format(mode=mode)
    subjects = os.listdir(raw_data_path)

    for subject in subjects:
        raw_data_mat[subject] = {}
        for r in roi:
            subjects_roi_path = os.path.join(raw_data_path, subject, f"{r}.pkl")
            raw_data_mat[subject][r] = pd.read_pickle()

    # Path to the input .pkl file
    pkl_file_path = "input_dataframe.pkl"

    # Load the DataFrame from the .pkl file
    df = pd.read_pickle(pkl_file_path)

    # Path to the output .mat file
    mat_file_path = "output_dataframe.mat"

    # Convert DataFrame to a dictionary
    data_dict = df.to_dict(orient='list')

    # Save the dictionary as a .mat file
    scipy.io.savemat(mat_file_path, data_dict)


def create_roi_data_to_pkl(mode: Mode, roi: list[str], sort_direction=""):
    raw_data_mat = {}
    raw_data_path = config.SUBNET_DATA_DF_DENORMALIZED.format(mode=mode.value)

    subjects = os.listdir(raw_data_path)
    subjects = [s for s in subjects if not s.startswith('.')]

    for subject in subjects:
        raw_data_mat[subject] = {}
        for r in roi:
            subjects_roi_path = os.path.join(raw_data_path, subject, f"{r}.pkl")
            raw_data_mat[subject][r] = pd.read_pickle(subjects_roi_path)

    # Save the dictionary to a pkl file
    file_name = f"{sort_direction}{len(subjects)}_subjects_{len(roi)}_roi_{mode.name.lower()}.pkl"
    with open(file_name, 'wb') as f:
        pickle.dump(raw_data_mat, f)


def create_networks_level_pkl(mode):
    raw_data = {}
    raw_data_mat = {}
    raw_data_path = config.NETWORK_SUBNET_DATA_DF_DENORMALIZED.format(mode=mode.value)
    networks = os.listdir(raw_data_path)
    networks = [n for n in networks if not n.startswith('.')]
    for net in networks:
        subjects_list = [s for s in os.listdir(f'{raw_data_path}/{net}') if not s.startswith('.')]
        for subject in subjects_list:
            subject = subject.replace('.pkl', '')
            raw_data_mat.setdefault(subject, {})
            subjects_net_path = os.path.join(raw_data_path, net, f"{subject}.pkl")
            raw_data_mat[subject][net] = pd.read_pickle(subjects_net_path)

        raw_data.update(raw_data_mat)

    # Save the dictionary to a pkl file
    with open(f'176_subjects_7_networks_{mode.name.lower()}.pkl', 'wb') as f:
        pickle.dump(raw_data_mat, f)


def save_activations_to_csv_pandas(roi_to_net_map, activation_results, csv_filename='activation_data.csv',
                                   window=None, method='mean'):
    # Create a list to hold the rows of data
    data = []

    # Loop through each ROI and network
    for roi, network in roi_to_net_map.items():
        roi_activations = activation_results.get(roi, {})

        # Handle window-based data or aggregate method
        if window is None:
            if method == 'mean':
                act_avg_values = [roi_activations[win]['avg'] for win in roi_activations]
                activation_value = sum(act_avg_values) / len(act_avg_values)
            else:
                act_max_values = [roi_activations[win]['avg'] for win in roi_activations]
                activation_value = max(act_max_values)

        elif window == "last_tr_movie":
            activation_value = roi_activations["avg"]
        else:
            activation_value = roi_activations.get(window, {}).get('avg', None)

        # Add the data to the list if activation value is available
        if activation_value is not None:
            data.append([network, roi, activation_value])

    # Convert the data into a pandas DataFrame
    df = pd.DataFrame(data, columns=['Network', 'ROI', 'Activation Value'])
    df = df.sort_values(by='Activation Value', ascending=False)

    # Save the DataFrame to a CSV file
    df.to_csv(csv_filename, index=False)

    print(f"Data successfully saved to {csv_filename}")


def list_all_objects(client, bucket_name, prefix):
    """
    List all object keys under a given prefix in an S3 bucket, handling pagination.

    Args:
        bucket_name (str): Name of the S3 bucket.
        prefix (str): Prefix to filter the objects.

    Returns:
        list[str]: A list of all object keys under the prefix.
    """
    keys = []
    continuation_token = None

    while True:
        # List objects with pagination
        list_kwargs = {'Bucket': bucket_name, 'Prefix': prefix}
        if continuation_token:
            list_kwargs['ContinuationToken'] = continuation_token

        response = client.list_objects_v2(**list_kwargs)

        # Extract keys from the current batch
        keys.extend([content['Key'] for content in response.get('Contents', [])])

        # Check if there are more keys to fetch
        if response.get('IsTruncated'):  # True if more results are available
            continuation_token = response['NextContinuationToken']
        else:
            break

    return keys


def create_roi_data_to_pkl_s3(mode: Mode, roi: list[str], sort_direction=""):
    s3_client = boto3.client('s3')
    bucket_name = "erezsimony"
    # Define the base S3 path for the raw data
    base_s3_path = f"Schaefer2018_SUBNET_{mode.name}_DF_denormalized"
    # Dictionary to store the aggregated data
    raw_data_mat = {}

    # Get the list of subjects from the S3 bucket
    try:

        all_keys = list_all_objects(s3_client, bucket_name, base_s3_path)
        subjects = list(set(key.split('/')[1] for key in all_keys if len(key.split('/')) == 3))
        print(f"Found subjects: {subjects}")
    except Exception as e:
        print(f"Error listing objects in S3: {e}")
        return

    # Fetch and process data for each subject and ROI
    for subject in tqdm.tqdm(subjects):
        raw_data_mat[subject] = {}
        for r in roi:
            roi_s3_path = f"{base_s3_path}/{subject}/{r}.pkl"
            try:
                # Fetch the ROI data from S3
                response = s3_client.get_object(Bucket=bucket_name, Key=roi_s3_path)
                roi_data = pickle.load(io.BytesIO(response['Body'].read()))
                raw_data_mat[subject][r] = roi_data
                print(f"Processed ROI {r} for subject {subject} in mode {mode.name} sorted {sort_direction}")
            except Exception as e:
                print(f"Error fetching ROI data {roi_s3_path} from S3: {e}")

    # Save the aggregated data to a .pkl file
    file_name = f"{sort_direction}_{len(subjects)}_subjects_{len(roi)}_roi_{mode.name.lower()}.pkl"
    with open(file_name, 'wb') as f:
        pickle.dump(raw_data_mat, f)

    # Upload the file to S3
    s3_output_path = f"output_data/{file_name}"
    try:
        s3_client.upload_file(file_name, bucket_name, s3_output_path)
        print(f"Aggregated data uploaded successfully to s3://{bucket_name}/{s3_output_path}")
    except Exception as e:
        print(f"Error uploading file to S3: {e}")


if __name__ == '__main__':
    StaticData.inhabit_class_members()
    rois = StaticData.ROI_NAMES
    with open("svc_all_rois_17_groups_10_sub_in_group_rest_between_data_5TR_window_activations_results.pkl", "rb") as f:
        decoding_scores = pickle.load(f)

    sorted_decoding_scores = sorted(decoding_scores.items(), key=lambda x: x[1]["13-18"]["avg"], reverse=True)
    sorted_rois = [roi[0] for roi in sorted_decoding_scores]
    highest_sorted_rois = sorted_rois[:10]
    lowest_sorted_rois = sorted_rois[-10:]

    #create_roi_data_to_pkl_s3(mode=Mode.REST, roi=highest_sorted_rois, sort_direction="highest_decding")
    create_roi_data_to_pkl_s3(mode=Mode.REST, roi=lowest_sorted_rois, sort_direction="lowest_decding")

    #create_roi_data_to_pkl_s3(mode=Mode.FIRST_REST_SECTION, roi=highest_sorted_rois, sort_direction="highest_decding")
    create_roi_data_to_pkl_s3(mode=Mode.FIRST_REST_SECTION, roi=lowest_sorted_rois, sort_direction="lowest_decding")

    #create_roi_data_to_pkl_s3(mode=Mode.TASK, roi=highest_sorted_rois, sort_direction="highest_decding")
    create_roi_data_to_pkl_s3(mode=Mode.TASK, roi=lowest_sorted_rois, sort_direction="lowest_decding")

