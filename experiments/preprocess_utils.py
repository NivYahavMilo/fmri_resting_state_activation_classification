import os

import pandas as pd

import config
from enums import PreprocessType, Mode


def _distances_preprocess_util(features):
    """
    Preprocesses the features by removing a specific value from a Series.

    Args:
        features (pd.Series): The Series containing the features to be preprocessed.

    Returns:
        precessed_features (pd.Series): The preprocessed features Series.
    """
    # Remove a specific value from the Series
    precessed_features = features.drop(features[features == 1].index)
    return precessed_features


def _preprocess(data: pd.DataFrame, preprocess_type: PreprocessType):
    """
    Preprocesses the data based on the given preprocess type.

    Args:
        data (pd.DataFrame): The DataFrame containing the data to be preprocessed.
        preprocess_type (PreprocessType): The type of preprocessing to be applied.

    Returns:
        preprocess_data (pd.DataFrame): The preprocessed data DataFrame.
    """
    preprocess_data = pd.DataFrame()
    for subject, modes in data.items():
        for mode, matrix_data in modes.items():

            for idx, movie_name in enumerate(matrix_data.columns):
                features = matrix_data[movie_name]
                if preprocess_type.DISTANCES:
                    features = _distances_preprocess_util(features)

                # Nothing special needs to be done when data type is activations.
                elif preprocess_type.ACTIVATIONS:
                    pass

                data = {'subject': subject, 'mode': mode, 'y': [movie_name]}
                for i, feature in enumerate(features):
                    data[i] = [feature]
                sample_df = pd.DataFrame(data)
                preprocess_data = pd.concat([preprocess_data, sample_df])

    preprocess_data = preprocess_data.reset_index(drop=True)
    return preprocess_data


def preprocess_dataset(roi: str, dataset: str, mode: Mode, preprocess_type: PreprocessType):
    """
    Preprocesses a dataset based on the specified mode and preprocess type.

    Args:
        dataset (str): Name of the dataset.
        mode (Mode): Mode enum value representing the desired mode (e.g., Mode.TRAIN, Mode.TEST).
        preprocess_type (PreprocessType): PreprocessType enum value representing the desired preprocess type.

    Returns:
        pd.DataFrame: Preprocessed dataset filtered based on the specified mode.
        :param preprocess_type: preprocess type Enum
        :param mode: Enum MODE
        :param dataset: dataset type
        :param roi: roi name

    """
    dataset_dir = os.path.join(config.PREPROCESSED_DATASETS_PATH, roi)
    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)

    dataset_path = os.path.join(dataset_dir, dataset)

    if os.path.isfile(dataset_path):
        preprocess_data = pd.read_pickle(dataset_path)
    else:
        raw_dataset_path = os.path.join(config.RAW_DATASETS_PATH, roi, dataset)
        data = pd.read_pickle(raw_dataset_path)
        preprocess_data = _preprocess(data, preprocess_type=preprocess_type)
        preprocess_data_path = os.path.join(dataset_dir, dataset)
        preprocess_data.to_pickle(preprocess_data_path)

    preprocess_data_mode = preprocess_data[preprocess_data['mode'] == mode.value.lower()]
    return preprocess_data_mode
