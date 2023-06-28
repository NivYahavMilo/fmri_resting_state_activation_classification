import os

import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC

import config
from enums import Mode, PreprocessType, FlowType
from experiments.train_utils import get_supervised_tensors

import scipy.io

def load_mat(file, var):
    # Load the .mat file
    mat_data = scipy.io.loadmat(file)

    # Extract the data from the loaded .mat file
    data = mat_data[var]  # Replace 'your_variable_name' with the actual variable name in the .mat file

    # Convert the data to a DataFrame
    df = pd.DataFrame()
    for g in range(data.shape[0]):
        df_g = pd.DataFrame(data[g, :, :])
        df = pd.concat([df, df_g])
    return df


def create_combined_modes_dataset(roi, preprocess_type: PreprocessType, data_type: FlowType):
    data_task = get_supervised_tensors(
        roi=roi,
        mode=Mode.TASK,
        preprocess_type=preprocess_type,
        rest_window=(13, 18),
        data_type=data_type
    )

    task_mat = load_mat(var='group_dist_task_all' ,file=r'C:\Users\nivy1\Documents\Engineering\MS.c\Thesis\fmri_resting_state_activation_classification\datasets\raw_datasets\RH_Default_pCunPCC_1\group_dist_task_all.mat')

    # Shuffle the DataFrame
    #data_task = data_task.sample(frac=1).reset_index(drop=True)

    y_task_train = data_task['y']
    X_task_train = data_task.drop(['y', 'group'], axis=1)
    X_task_train = X_task_train.apply(scipy.stats.zscore)



    data_rest = get_supervised_tensors(
        roi=roi,
        mode=Mode.REST,
        preprocess_type=preprocess_type,
        rest_window=(13, 18),
        data_type=data_type
    )
    rest_mat = load_mat(var='group_dist_rest_all' ,file=r'C:\Users\nivy1\Documents\Engineering\MS.c\Thesis\fmri_resting_state_activation_classification\datasets\raw_datasets\RH_Default_pCunPCC_1\group_dist_rest_all.mat')

    data_rest = data_rest.sample(frac=1).reset_index(drop=True)
    y_rest_test = data_rest['y']
    X_rest_test = data_rest.drop(['y', 'group'], axis=1)
    X_rest_test = X_rest_test.apply(scipy.stats.zscore)

    return X_task_train, y_task_train, X_rest_test, y_rest_test


def train(roi: str, preprocess_type: PreprocessType, data_type: FlowType):
    X_task, y_task, X_rest, y_rest = create_combined_modes_dataset(roi, preprocess_type, data_type)
    # Encode the movie names (labels) using LabelEncoder
    label_encoder = LabelEncoder()
    y_task_encoded = label_encoder.fit_transform(y_task)
    y_rest_encoded = label_encoder.fit_transform(y_rest)

    model_svm = make_pipeline(
        SVC()  # SVM classifier
    )

    model_svm.fit(X_task, y_task_encoded)
    svm_accuracy = model_svm.score(X_rest, y_rest_encoded)

    model_knn = make_pipeline(
        KNeighborsClassifier(n_neighbors=3)  # SVM classifier
    )

    model_knn.fit(X_task, y_task_encoded)
    knn_accuracy = model_knn.score(X_rest, y_rest_encoded)

    print('evaluating on:', len(X_rest), 'samples')

    # Evaluate the classifier on the testing set

    # Generate the classification report
    y_test_tags = label_encoder.inverse_transform(y_rest_encoded)
    knn_y_pred_tags = label_encoder.inverse_transform(model_knn.predict(X_rest))
    svm_y_pred_tags = label_encoder.inverse_transform(model_svm.predict(X_rest))

    knn_report = classification_report(y_test_tags, knn_y_pred_tags)
    svm_report = classification_report(y_test_tags, svm_y_pred_tags)

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

    # Specify the file path where you want to save the report
    res_dir_path = os.path.join(config.EXPERIMENTS_RESULTS, roi)
    if not os.path.exists(res_dir_path):
        os.makedirs(res_dir_path)

    train_file_name = f'{preprocess_type.value.lower()}_{data_type.value.lower()}_train_on_movie_predict_on_rest_remove_1'
    file_path = os.path.join(res_dir_path, f'{train_file_name}_classification_report.txt')

    # Open the file in write mode and save the classification report
    combined_str_reports = f"Average Accuracy SVM: {svm_accuracy}\n\n" \
                           f"Classification Reports SVM:\n {svm_report}\n\n\n" \
                           f"Average Accuracy KNN: {knn_accuracy}\n\n\n" \
                           f"Classification Reports KNN:\n {knn_report}\n\n\n"

    with open(file_path, 'w') as file:
        file.write(f"{combined_str_reports}")


if __name__ == '__main__':
    train(roi='RH_Default_pCunPCC_1', preprocess_type=PreprocessType.DISTANCES, data_type=FlowType.GROUP_SUBJECTS)
    #train(roi='RH_DorsAttn_Post_2', preprocess_type=PreprocessType.ACTIVATIONS, data_type=FlowType.GROUP_SUBJECTS)
    #train(roi='RH_Vis_18', preprocess_type=PreprocessType.ACTIVATIONS, data_type=FlowType.GROUP_SUBJECTS)
