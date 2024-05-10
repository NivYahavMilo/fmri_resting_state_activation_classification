import os
import pickle

import pandas as pd
import scipy.io

import config
from enums import Mode


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
    with open(f'176_subjects_8_roi_{mode.name.lower()}.pkl', 'wb') as f:
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


if __name__ == '__main__':
    create_roi_data_to_pkl(mode=Mode.FIRST_REST_SECTION, roi=[
        'RH_Default_Par_1', 'RH_Default_PFCdPFCm_9', 'LH_SalVentAttn_FrOperIns_1', 'RH_Default_PFCdPFCm_6',
        'RH_Default_PFCv_3', 'RH_Default_Temp_6', 'RH_SalVentAttn_FrOperIns_1', 'RH_Limbic_TempPole_6'
    ])

    # create_networks_level_pkl(Mode.FIRST_REST_SECTION)
    # create_networks_level_pkl(Mode.REST)
    # create_networks_level_pkl(Mode.TASK)
