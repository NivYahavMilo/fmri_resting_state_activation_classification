import os

import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC

import config
from enums import Mode, PreprocessType


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

    dataset_path = os.path.join(config.PREPROCESSED_DATASETS_PATH, roi, dataset)
    if os.path.isfile(dataset_path):
        preprocess_data = pd.read_pickle(dataset_path)
    else:
        raw_dataset_path = os.path.join(config.RAW_DATASETS_PATH, roi, dataset)
        data = pd.read_pickle(raw_dataset_path)
        preprocess_data = _preprocess(data, preprocess_type=preprocess_type)

        preprocess_data.to_pickle(dataset_path)

    preprocess_data_mode = preprocess_data[preprocess_data['mode'] == mode.value.lower()]
    return preprocess_data_mode


def train():
    mode = Mode.REST
    roi = 'RH_DorsAttn_Post_2'
    data_type = PreprocessType.DISTANCES
    dataset_name = f'{data_type.value.lower()}_176_subjects_RH_DorsAttn_Post_2_roi_(0, 10)_rest_window'
    dataset = preprocess_dataset(
        roi=roi,
        dataset=f'{dataset_name}.pkl',
        mode=mode,
        preprocess_type=data_type
    )
    # Remove columns that don't represent movie's features
    y = dataset['y'].tolist()
    X = dataset.drop(['subject', 'mode', 'y'], axis=1).values
    # Encode the movie names (labels) using LabelEncoder
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, stratify=y, shuffle=True, test_size=0.2,
                                                        random_state=0)

    print('training on:', len(X_train), 'samples')
    # Train an SVM classifier on the training set
    svm_classifier = LinearSVC()
    svm_classifier.fit(X_train, y_train)

    print('evaluating on:', len(X_test), 'samples')

    # Evaluate the classifier on the testing set
    accuracy = svm_classifier.score(X_test, y_test)

    # Generate the classification report
    y_pred = svm_classifier.predict(X_test)
    y_test_tags = label_encoder.inverse_transform(y_test)
    y_pred_tags = label_encoder.inverse_transform(y_pred)
    report = classification_report(y_test_tags, y_pred_tags)

    print("Accuracy:", accuracy)
    print("Classification Report:\n")
    print(report)

    # Specify the file path where you want to save the report
    file_path = os.path.join(config.EXPERIMENTS_RESULTS, roi,
                             f'{dataset_name}_MODE_{mode.name}_classification_report.txt')

    # Open the file in write mode and save the classification report
    with open(file_path, 'w') as file:
        file.write(f"Accuracy: {accuracy}\nClassification Report:\n {report}")


if __name__ == '__main__':
    train()
