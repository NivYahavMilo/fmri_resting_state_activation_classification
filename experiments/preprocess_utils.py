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
    # precessed_features = features.apply(lambda feature: 1 - feature)
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

    raw_dataset_path = os.path.join(config.RAW_DATASETS_PATH, roi, dataset)
    data = pd.read_pickle(raw_dataset_path)
    preprocess_data = _preprocess(data, preprocess_type=preprocess_type)
    preprocess_data_mode = preprocess_data[preprocess_data['mode'] == mode.value.lower()]
    return preprocess_data_mode


def dissimilarity_preprocessing(movie_df: pd.DataFrame, rest_df: pd.DataFrame):
    # Filter the movie_df to include only the 5 groups for the training set
    train_df = movie_df[movie_df['group'].isin(range(1, 6))].reset_index(drop=True)
    train_df_labels = train_df['y'].values
    train_df = train_df.drop(['group', 'y'], axis=1)

    # Filter the rest_df to include only the samples from group 6
    # Create an empty DataFrame for the test set

    test_sample_movies = movie_df[movie_df['group'] == 6].drop(['group'], axis=1)
    test_sample_movies = test_sample_movies.drop(['y'], axis=1)

    test_df = rest_df[rest_df['group'] == 6].reset_index(drop=True)
    test_df_labels = test_df['y'].values
    test_df = test_df.drop(['group', 'y'], axis=1)

    # Perform correlation between test samples and train samples in group 6
    test_set = {}
    test_set_df = pd.DataFrame()

    for rest_i in range(len(test_df)):
        for movie_i in range(len(test_sample_movies)):
            feature_df = pd.DataFrame(
                {'rest_test': test_df.iloc[rest_i], 'movie_test': test_sample_movies.iloc[movie_i]}).corr()

            feature = 1 - feature_df.loc['rest_test'].at['movie_test']
            test_set[movie_i] = [feature]

        test_set_df = pd.concat([test_set_df, pd.DataFrame(test_set)])

    test_set_df = test_set_df.reset_index(drop=True)
    return train_df, train_df_labels, test_set_df, test_df_labels
