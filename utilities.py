import os
import pickle

import pandas as pd
import scipy.io

import config
from enums import Mode
from  static_data.static_data import StaticData

StaticData.inhabit_class_members()


def get_clip_index_mapping(inverse: bool = False):
    clip_mapper = {
        "0": "testretest",
        "1": "twomen",
        "2": "bridgeville",
        "3": "pockets",
        "4": "overcome",
        "5": "inception",
        "6": "socialnet",
        "7": "oceans",
        "8": "flower",
        "9": "hotel",
        "10": "garden",
        "11": "dreary",
        "12": "homealone",
        "13": "brokovich",
        "14": "starwars"
    }

    if inverse:
        clip_mapper = {v: k for k, v in clip_mapper.items()}

    return clip_mapper


def convert_pkl_to_mat(mode: Mode, roi: list[str]):
    raw_data_mat = {}
    raw_data_path = config.SUBNET_DATA_DF.format(mode=mode)
    subjects = os.listdir(raw_data_path)

    for subject in subjects:
        raw_data_mat[subject] = {}
        for r in roi:
            subjects_roi_path = os.path.join(raw_data_path, subject, f"{r}.pkl")
            raw_data_mat[subject][r] = pd.read_pickle()

    # Path to the input .pkl file
    pkl_file_path = "input_dataframe.pkl"

    # Load the DataFrame from the .pkl file
    df = pd.read_pickle(pkl_file_path)

    # Path to the output .mat file
    mat_file_path = "output_dataframe.mat"

    # Convert DataFrame to a dictionary
    data_dict = df.to_dict(orient='list')

    # Save the dictionary as a .mat file
    scipy.io.savemat(mat_file_path, data_dict)


def create_roi_data_to_pkl(mode: Mode, roi: list[str]):
    raw_data_mat = {}
    raw_data_path = config.SUBNET_DATA_DF_DENORMALIZED.format(mode=mode.value)

    subjects = os.listdir(raw_data_path)
    subjects = [s for s in subjects if not s.startswith('.')]

    for subject in subjects:
        raw_data_mat[subject] = {}
        for r in roi:
            subjects_roi_path = os.path.join(raw_data_path, subject, f"{r}.pkl")
            raw_data_mat[subject][r] = pd.read_pickle(subjects_roi_path)

    # Save the dictionary to a pkl file
    with open(f'176_subjects_{len(roi)}_roi_{mode.name.lower()}.pkl', 'wb') as f:
        pickle.dump(raw_data_mat, f)


def create_networks_level_pkl(mode):
    raw_data = {}
    raw_data_mat = {}
    raw_data_path = config.NETWORK_SUBNET_DATA_DF_DENORMALIZED.format(mode=mode.value)
    networks = os.listdir(raw_data_path)
    networks = [n for n in networks if not n.startswith('.')]
    for net in networks:
        subjects_list = [s for s in os.listdir(f'{raw_data_path}/{net}') if not s.startswith('.')]
        for subject in subjects_list:
            subject = subject.replace('.pkl', '')
            raw_data_mat.setdefault(subject, {})
            subjects_net_path = os.path.join(raw_data_path, net, f"{subject}.pkl")
            raw_data_mat[subject][net] = pd.read_pickle(subjects_net_path)

        raw_data.update(raw_data_mat)

    # Save the dictionary to a pkl file
    with open(f'176_subjects_7_networks_{mode.name.lower()}.pkl', 'wb') as f:
        pickle.dump(raw_data_mat, f)


def save_activations_to_csv_pandas(roi_to_net_map, activation_results, csv_filename='activation_data.csv',
                                   window=None, method='mean'):
    # Create a list to hold the rows of data
    data = []

    # Loop through each ROI and network
    for roi, network in roi_to_net_map.items():
        roi_activations = activation_results.get(roi, {})

        # Handle window-based data or aggregate method
        if window is None:
            if method == 'mean':
                act_avg_values = [roi_activations[win]['avg'] for win in roi_activations]
                activation_value = sum(act_avg_values) / len(act_avg_values)
            else:
                act_max_values = [roi_activations[win]['avg'] for win in roi_activations]
                activation_value = max(act_max_values)
        else:
            activation_value = roi_activations.get(window, {}).get('avg', None)

        # Add the data to the list if activation value is available
        if activation_value is not None:
            data.append([network, roi, activation_value])

    # Convert the data into a pandas DataFrame
    df = pd.DataFrame(data, columns=['Network', 'ROI', 'Activation Value'])
    df = df.sort_values(by='Activation Value', ascending=False)

    # Save the DataFrame to a CSV file
    df.to_csv(csv_filename, index=False)

    print(f"Data successfully saved to {csv_filename}")

if __name__ == '__main__':
    # activation_res = pd.read_pickle("all_rois_groups_activations_results.pkl")
    # first_window = "0-5"
    # save_activations_to_csv_pandas(roi_to_net_map=StaticData.ROI_TO_NETWORK_MAPPING,
    #                                activation_results=activation_res,
    #                                csv_filename=f'activation_data_window_{first_window}.csv',
    #                             window=first_window)
    # last_window = "13-18"
    # save_activations_to_csv_pandas(roi_to_net_map=StaticData.ROI_TO_NETWORK_MAPPING,
    #                                activation_results=activation_res,
    #                                csv_filename=f'activation_data_window_{last_window}.csv',
    #                             window=last_window)

    create_roi_data_to_pkl(mode=Mode.TASK, roi=['RH_Vis_16', 'RH_Vis_5', 'LH_Vis_15' , 'RH_SomMot_2', 'LH_SomMot_3'])



    # create_networks_level_pkl(Mode.FIRST_REST_SECTION)
    # create_networks_level_pkl(Mode.REST)
    # create_networks_level_pkl(Mode.TASK)
