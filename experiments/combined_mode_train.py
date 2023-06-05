import os

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC

import config
from enums import Mode, PreprocessType, FlowType
from experiments.train_utils import get_supervised_tensors


def create_combined_modes_dataset(roi, preprocess_type: PreprocessType, data_type: FlowType):
    X_task, y_task = get_supervised_tensors(
        roi=roi,
        mode=Mode.TASK,
        preprocess_type=preprocess_type,
        rest_window=(13, 18),
        data_type=data_type
    )

    X_rest, y_rest = get_supervised_tensors(
        roi=roi,
        mode=Mode.REST,
        preprocess_type=preprocess_type,
        rest_window=(13, 18),
        data_type=data_type
    )

    return X_task, X_rest, y_task, y_rest


def train(roi: str, preprocess_type: PreprocessType, data_type: FlowType):
    X_task, X_rest, y_task, y_rest = create_combined_modes_dataset(roi, preprocess_type, data_type)
    # Encode the movie names (labels) using LabelEncoder
    label_encoder = LabelEncoder()
    y_task_encoded = label_encoder.fit_transform(y_task)
    y_rest_encoded = label_encoder.fit_transform(y_rest)

    print('training on:', len(X_task), 'samples')
    # Train an SVM classifier on the training set
    svm_classifier = LinearSVC()
    svm_classifier.fit(X_task, y_task_encoded)

    knn = KNeighborsClassifier(n_neighbors=3)
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

    train_file_name = f'{preprocess_type.value.lower()}_{data_type.value.lower()}_train_on_movie_predict_on_rest'
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
    train(roi='RH_DorsAttn_Post_2', preprocess_type=PreprocessType.DISTANCES, data_type=FlowType.GROUP_SUBJECTS)
    # train(roi='RH_Default_pCunPCC_1', preprocess_type=PreprocessType.DISTANCES, data_type=FlowType.SINGLE_SUBJECT)
    # train(roi='RH_Vis_18', preprocess_type=PreprocessType.ACTIVATIONS, data_type=FlowType.SINGLE_SUBJECT)
    # train(roi='RH_Default_pCunPCC_1', preprocess_type=PreprocessType.ACTIVATIONS, data_type=FlowType.SINGLE_SUBJECT)
