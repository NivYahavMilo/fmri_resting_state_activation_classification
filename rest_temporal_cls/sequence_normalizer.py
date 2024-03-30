import numpy as np
import pandas as pd
from tqdm import tqdm

from dataloader.dataloader import DataLoader
from enums import Mode
from rest_temporal_cls.utils import Utils


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


def get_normalized_data(roi: str, group_average: bool, first_rest: bool = False, **kwargs):
    data_loader = DataLoader()
    normalized_subjects_data = pd.DataFrame()
    subjects = Utils.subject_list
    print("Normalizing Data")
    for subject in tqdm(subjects):
        first_rest_sequence = pd.DataFrame()
        if first_rest:
            first_rest_sequence = data_loader.load_single_subject_activations(roi, subject, Mode.FIRST_REST_SECTION)

        clip_sequence = data_loader.load_single_subject_activations(roi, subject, Mode.TASK)
        clip_sequence['is_rest'] = 0
        rest_sequence = data_loader.load_single_subject_activations(roi, subject, Mode.REST)
        rest_sequence['is_rest'] = 1
        norm_sub_data = z_score_concatenated_scan(clip_sequence, rest_sequence)
        normalized_subjects_data = pd.concat([normalized_subjects_data, norm_sub_data])

    return normalized_subjects_data
