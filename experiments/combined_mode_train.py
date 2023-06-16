import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC

import config
from enums import Mode, PreprocessType, FlowType
from experiments.train_utils import get_supervised_tensors


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


def create_combined_modes_dataset(roi, preprocess_type: PreprocessType, data_type: FlowType):
    data_task = get_supervised_tensors(
        roi=roi,
        mode=Mode.TASK,
        preprocess_type=preprocess_type,
        rest_window=(13, 18),
        data_type=data_type
    )
    # Shuffle the DataFrame
    data_task = data_task.sample(frac=1).reset_index(drop=True)
    y_task_train = data_task['y']
    X_task_train = data_task.drop(['y', 'group'], axis=1)
    precessed_features = features.drop(features[features == 1].index)
    # X_task_train = X_task_train.apply(lambda x: -1*x)

    data_rest = get_supervised_tensors(
        roi=roi,
        mode=Mode.REST,
        preprocess_type=preprocess_type,
        rest_window=(13, 18),
        data_type=data_type
    )

    data_rest = data_rest.sample(frac=1).reset_index(drop=True)
    y_rest_test = data_rest['y']
    X_rest_test = data_rest.drop(['y', 'group'], axis=1)
    # X_rest_test = X_rest_test.apply(lambda x: -1*x)

    return X_task_train, y_task_train, X_rest_test, y_rest_test


def train(roi: str, preprocess_type: PreprocessType, data_type: FlowType):
    X_task, y_task, X_rest, y_rest = create_combined_modes_dataset(roi, preprocess_type, data_type)
    # Encode the movie names (labels) using LabelEncoder
    label_encoder = LabelEncoder()
    y_task_encoded = label_encoder.fit_transform(y_task)
    y_rest_encoded = label_encoder.fit_transform(y_rest)

    print('training on:', len(X_task), 'samples')
    # Train an SVM classifier on the training set
    svm_classifier = LinearSVC()
    svm_classifier.fit(X_task, y_task_encoded)

    knn = KNeighborsClassifier(n_neighbors=1)
    knn.fit(X_task, y_task_encoded)

    lda = LinearDiscriminantAnalysis()
    lda.fit(X_task, y_task_encoded)

    print('evaluating on:', len(X_rest), 'samples')

    # Evaluate the classifier on the testing set

    knn_accuracy = knn.score(X_rest, y_rest_encoded)
    knn_y_pred = knn.predict(X_rest)

    svm_accuracy = svm_classifier.score(X_rest, y_rest_encoded)
    svm_y_pred = svm_classifier.predict(X_rest)

    lda_accuracy = lda.score(X_rest, y_rest_encoded)
    lda_y_pred = lda.predict(X_rest)

    # Generate the classification report
    y_test_tags = label_encoder.inverse_transform(y_rest_encoded)
    knn_y_pred_tags = label_encoder.inverse_transform(knn_y_pred)
    svm_y_pred_tags = label_encoder.inverse_transform(svm_y_pred)
    lda_y_pred_tags = label_encoder.inverse_transform(lda_y_pred)

    knn_report = classification_report(y_test_tags, knn_y_pred_tags)
    svm_report = classification_report(y_test_tags, svm_y_pred_tags)
    lda_report = classification_report(y_test_tags, lda_y_pred_tags)

    # Create confusion matrix
    cm = confusion_matrix(y_test_tags.tolist(), svm_y_pred_tags.tolist())

    # Create ConfusionMatrixDisplay object
    cm_display = ConfusionMatrixDisplay(confusion_matrix=cm)

    # Display confusion matrix
    fig, ax = plt.subplots(figsize=(8, 6))
    cm_display.plot(ax=ax, cmap='Blues')
    plt.title('SVM Confusion Matrix')
    plt.show()

    print('Average Accuracy SVM:', svm_accuracy)
    print('Classification Reports SVM:')
    print(svm_report)

    print('Average Accuracy KNN:', knn_accuracy)
    print('Classification Reports KNN:')
    print(knn_report)

    print('Average Accuracy LDA:', lda_accuracy)
    print('Classification Reports LDA:')
    print(lda_report)

    # Specify the file path where you want to save the report
    res_dir_path = os.path.join(config.EXPERIMENTS_RESULTS, roi)
    if not os.path.exists(res_dir_path):
        os.makedirs(res_dir_path)

    train_file_name = f'{preprocess_type.value.lower()}_{data_type.value.lower()}_train_on_movie_predict_on_rest_dissimilarity_approach'
    file_path = os.path.join(res_dir_path, f'{train_file_name}_classification_report.txt')

    # Open the file in write mode and save the classification report
    combined_str_reports = f"Average Accuracy SVM: {svm_accuracy}\n\n" \
                           f"Classification Reports SVM:\n {svm_report}\n\n\n" \
                           f"Average Accuracy KNN: {knn_accuracy}\n\n\n" \
                           f"Classification Reports KNN:\n {knn_report}\n\n\n" \
                           f"Average Accuracy LDA: {lda_accuracy}\n\n\n" \
                           f"Classification Reports LDA:\n {lda_report}"

    with open(file_path, 'w') as file:
        file.write(f"{combined_str_reports}")


if __name__ == '__main__':
    train(roi='RH_Default_pCunPCC_1', preprocess_type=PreprocessType.DISTANCES, data_type=FlowType.GROUP_SUBJECTS)
    # train(roi='RH_Default_pCunPCC_1', preprocess_type=PreprocessType.DISTANCES, data_type=FlowType.SINGLE_SUBJECT)
    # train(roi='RH_Vis_18', preprocess_type=PreprocessType.ACTIVATIONS, data_type=FlowType.SINGLE_SUBJECT)
    # train(roi='RH_Default_pCunPCC_1', preprocess_type=PreprocessType.ACTIVATIONS, data_type=FlowType.SINGLE_SUBJECT)
