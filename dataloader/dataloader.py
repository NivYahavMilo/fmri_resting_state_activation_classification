import os
import pickle
from typing import Tuple

import pandas as pd
from scipy.stats import zscore

import config
from enums import Mode, PreprocessType, FlowType
from utils import get_clip_index_mapping


class DataLoader:

    def _group_subjects_load(self, roi: str, group_size: int, group_i: int, mode: Mode):
        data_mode = config.SUBNET_AVG_N_SUBJECTS.format(
            mode=mode.value,
            n_subjects=group_size,
            group_i=group_i
        )
        data_path = os.path.join(data_mode, f'{roi}.pkl')
        data_df = pd.read_pickle(data_path)
        return data_df

    def load_single_subject_activations(self, roi: str, subject: str, mode: Mode) -> pd.DataFrame:
        """
        Load activations data for a single subject.

        Args:
            roi (str): Region of interest.
            subject (str): Subject identifier.
            mode (Mode): Data mode (REST or TASK).

        Returns:
            pd.DataFrame: Activations data for the specified ROI and subject.
        """
        data_mode = config.SUBNET_DATA_DF.format(mode=mode.value)
        data_path = os.path.join(data_mode, subject, f'{roi}.pkl')
        data_df = pd.read_pickle(data_path)
        return data_df

    def get_rest_subsequence_window(self, data: pd.DataFrame, rest_window: Tuple[int, int]) -> pd.DataFrame:
        """
        Filter the DataFrame based on the specified time window for REST data.

        Args:
            data (pd.DataFrame): Activations data.
            rest_window (Tuple[int, int]): REST time window (start_time, end_time).

        Returns:
            pd.DataFrame: Filtered DataFrame within the specified REST time window.
        """
        start_time, end_time = rest_window

        # Filter the DataFrame based on the time window
        filtered_df = data[(data['timepoint'] >= start_time) & (data['timepoint'] <= end_time)]
        return filtered_df

    def get_task_subsequence(self, data: pd.DataFrame, window_size: int) -> pd.DataFrame:
        """
        Get the last window_size timepoints for each movie in the DataFrame.

        Args:
            data (pd.DataFrame): Activations data.
            window_size (int): Size of the task window.

        Returns:
            pd.DataFrame: Filtered DataFrame with the last window_size timepoints for each movie.
        """
        # Group the DataFrame by movie index and filter the last window size timepoints for each movie
        filtered_df = data.groupby('y').apply(lambda x: x.iloc[-window_size:])

        # Reset the index of the filtered DataFrame
        filtered_df = filtered_df.reset_index(drop=True)

        return filtered_df

    def _mean_subsequence(self, sequence: pd.DataFrame, preprocess_type: PreprocessType, z_score: bool) -> pd.DataFrame:
        """
        Calculate the mean along the feature axis for a given DataFrame sequence.

        Args:
            sequence (pd.DataFrame): DataFrame sequence.
            z_score (bool): Flag to indicate whether to apply z-score normalization.

        Returns:
            pd.DataFrame: Transposed DataFrame with mean values along the feature axis.
        """
        feature_columns = [col for col in sequence.columns if ('feat' in str(col)) or (isinstance(col, int))]

        # Calculate the mean along the feature axis
        mean_df = sequence.groupby('y')[feature_columns].mean()

        # Transpose the DataFrame so that each movie becomes a column
        transposed_df = mean_df.transpose()

        if z_score:
            # Apply Z-score along each column (movie)
            transposed_df = transposed_df.apply(zscore)

        # Rename the movie ID columns using the mapping dictionary
        clip_name_mapping = get_clip_index_mapping(inverse=True)
        transposed_df.columns = clip_name_mapping.keys()

        # Drop 'testretest' column if it exists
        transposed_df = transposed_df.drop('testretest', axis=1)

        # create distances matrix
        if preprocess_type == PreprocessType.DISTANCES:
            transposed_df = transposed_df.corr()

        return transposed_df

    def export_to_pkl(self, data: dict, **params):
        """
        Export the data dictionary to a pickle file.

        Args:
            data (dict): Data dictionary to be exported.
            params (dict): Preprocess parameters.

        Returns:
            None
        """
        # Specify the file name with the preprocess properties.
        roi: str = params['roi']
        file_name = f'{params["preprocess_type"].value}' \
                    f'_{params["group_size"]}_subjects_group_size_' \
                    f'{roi}_roi_' \
                    f'{str(params["rest_window"])}_rest_window.pkl'

        # Specify the file path where you want to save the pickle file
        file_path = os.path.join(config.RAW_DATASETS_PATH, roi, file_name)

        # Open the file in binary mode and save the dictionary as a pickle file
        with open(file_path, 'wb') as file:
            pickle.dump(data, file)

    def preprocess(self, flow_type: FlowType):
        """
        Perform the preprocessing steps for all subjects and export the results to a pickle file.

        Returns:
            None
        """
        preprocess_data = {}
        preprocess_params = {
            'roi': 'RH_DorsAttn_Post_2',
            'subject': '',
            'rest_window': (10, 18),
            'task_window_size': 10,
            'k_subjects': 176,
            'group_size': 35,
            'mode': None,
            'preprocess_type': PreprocessType.DISTANCES
        }

        if flow_type == FlowType.SINGLE_SUBJECT:
            preprocess_data = self.single_subject_flow(**preprocess_params)
        elif flow_type == FlowType.GROUP_SUBJECTS:
            preprocess_data = self.group_subjects_flow(**preprocess_params)

        self.export_to_pkl(data=preprocess_data, **preprocess_params)

    def group_subjects_flow(self, **preprocess_params):
        preprocess_data = {}

        preprocess_params.pop('k_subjects')
        preprocess_params.pop('subject')
        rest_window: Tuple = preprocess_params.pop('rest_window')
        task_window: int = preprocess_params.pop('task_window_size')
        preprocess_type: PreprocessType = preprocess_params.pop('preprocess_type')
        n_groups: int = preprocess_params.pop('n_groups', 6)

        for group_i in range(1, n_groups+1):
            # Rest flow
            preprocess_params['mode'] = Mode.REST
            preprocess_params['group_i'] = group_i
            data = self._group_subjects_load(**preprocess_params)
            subsequence_data = self.get_rest_subsequence_window(data, rest_window)
            subsequence_mean_rest = self._mean_subsequence(
                sequence=subsequence_data,
                preprocess_type=preprocess_type,
                z_score=True
            )

            # Task flow
            preprocess_params['mode'] = Mode.TASK
            data = self._group_subjects_load(**preprocess_params)
            subsequence_data = self.get_task_subsequence(data, task_window)
            subsequence_mean_task = self._mean_subsequence(
                sequence=subsequence_data,
                preprocess_type=preprocess_type,
                z_score=True
            )

            mean_modes_sequences = {'task': subsequence_mean_task, 'rest': subsequence_mean_rest}
            preprocess_data[group_i] = mean_modes_sequences

        return preprocess_data

    def single_subject_flow(self, **preprocess_params):
        """
        Perform the preprocessing flow.

        Returns:
            dict: Dictionary containing the mean sequences for task and rest modes.
        """

        preprocess_params.pop('group_size')
        preprocess_data = {}
        subjects = os.listdir(config.SUBNET_DATA_DF.format(mode=Mode.TASK.value))
        for subject in subjects:
            preprocess_params['subject'] = subject

            subjects_amount: int = preprocess_params.pop('k_subjects')
            rest_window: Tuple = preprocess_params.pop('rest_window')
            task_window: int = preprocess_params.pop('task_window_size')
            preprocess_type: PreprocessType = preprocess_params.pop('preprocess_type')

            # Rest flow
            preprocess_params['mode'] = Mode.REST
            data = self.load_single_subject_activations(**preprocess_params)
            subsequence_data = self.get_rest_subsequence_window(data, rest_window)
            subsequence_mean_rest = self._mean_subsequence(
                sequence=subsequence_data,
                preprocess_type=preprocess_type,
                z_score=True
            )

            # Task flow
            preprocess_params['mode'] = Mode.TASK
            data = self.load_single_subject_activations(**preprocess_params)
            subsequence_data = self.get_task_subsequence(data, task_window)
            subsequence_mean_task = self._mean_subsequence(
                sequence=subsequence_data,
                preprocess_type=preprocess_type,
                z_score=True
            )

            mean_modes_sequences = {'task': subsequence_mean_task, 'rest': subsequence_mean_rest}
            preprocess_data[subject] = mean_modes_sequences

        return preprocess_data


if __name__ == '__main__':
    dataloader = DataLoader()
    dataloader.preprocess(flow_type=FlowType.GROUP_SUBJECTS)
