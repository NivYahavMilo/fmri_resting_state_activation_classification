import os

from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC

import config
from enums import Mode, PreprocessType
from experiments.preprocess_utils import preprocess_dataset


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
    svm_classifier = SVC()
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
