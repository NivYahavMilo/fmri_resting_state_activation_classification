import concurrent.futures
import os

import boto3
import numpy as np
import pandas as pd
from tqdm import tqdm

from aws_utils.fetch_s3 import fetch_object_from_s3
from aws_utils.upload_s3 import s3_client
from dataloader.dataloader import DataLoader
from enums import Mode
from utils import Utils


def _z_score(seq: np.array, axis: int) -> np.array:
    """
    Computes the Z-score normalization of a given sequence.

    Parameters:
    seq (numpy array): Input sequence to normalize.
    axis:
        0 -> columns normalization
        1 -> rows normalization

    Returns:
    numpy array: Normalized sequence.
    """
    # Compute the Z-score normalization
    seq = (1 / np.std(seq, axis=axis)) * (seq - np.mean(seq, axis=axis))
    return seq


def z_score_concatenated_scan(clip_sequence, rest_sequence):
    concat_scans = pd.DataFrame()
    for scan, clips in Utils.movie_scan_mapping.items():
        zs_df = pd.DataFrame()
        for clip in clips:
            clip_seq = clip_sequence[clip_sequence['y'] == clip]
            rest_seq = rest_sequence[rest_sequence['y'] == clip]
            concat_clip_df = pd.concat([clip_seq, rest_seq])

            zs_df = pd.concat([zs_df, concat_clip_df])

        static_columns = ['y', 'timepoint', 'Subject', 'is_rest']
        dropped_columns = zs_df[static_columns]
        zs_df_cp = zs_df.copy()
        zs_df_cp = zs_df_cp.drop(static_columns, axis=1)
        zs_df_cp = zs_df_cp.apply(lambda x: _z_score(x, axis=0))
        zs_df_cp[static_columns] = dropped_columns

        concat_scans = pd.concat([concat_scans, zs_df_cp])

    return concat_scans


def get_normalized_data(roi: str, first_rest: bool = False, resting_state: bool = False):
    s3 = boto3.client('s3')
    bucket_name = 'erezsimony'
    s3_destination_path = f'processed_rois{"_resting_state" if resting_state else ""}/{roi}.pkl'
    if os.path.exists(s3_destination_path):
        print(f"Found {s3_destination_path} locally")
        return pd.read_pickle(s3_destination_path)

    normalized_subjects_data = fetch_object_from_s3(s3_client, bucket_name, s3_destination_path)
    if normalized_subjects_data is not None:
        return normalized_subjects_data

    data_loader = DataLoader()
    normalized_subjects_data = pd.DataFrame()
    subjects = Utils.subject_list.copy()
    subjects.remove('111312')

    def process_subject(subject):
        first_rest_sequence = pd.DataFrame()
        if first_rest:
            first_rest_sequence = data_loader.load_single_subject_activations(roi, subject, Mode.FIRST_REST_SECTION)

        clip_sequence = data_loader.load_single_subject_activations(
            roi, subject, Mode.TASK if not resting_state else Mode.RESTING_STATE_TASK
        )
        clip_sequence['is_rest'] = 0

        rest_sequence = data_loader.load_single_subject_activations(
            roi, subject, Mode.REST if not resting_state else Mode.RESTING_STATE_REST
        )
        rest_sequence['is_rest'] = 1

        norm_sub_data = z_score_concatenated_scan(clip_sequence, rest_sequence)
        return norm_sub_data

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = {executor.submit(process_subject, subject): subject for subject in subjects}
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(subjects)):
            subject_data = future.result()
            normalized_subjects_data = pd.concat([normalized_subjects_data, subject_data])


    normalized_subjects_data.to_pickle(s3_destination_path)

    try:
        s3.upload_file(s3_destination_path, bucket_name, s3_destination_path)
        print(f"File {s3_destination_path} uploaded successfully to {s3_destination_path}")

    except Exception as e:
        print(f"Failed to upload {s3_destination_path} to S3: {e}")

    return normalized_subjects_data
