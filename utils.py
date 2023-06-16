import os

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


if __name__ == '__main__':
    pass
