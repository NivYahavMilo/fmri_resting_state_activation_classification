import os
import pickle
from typing import Callable

import boto3
import hcp_utils as hcp
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
from nilearn.plotting import plot_surf_stat_map
from tqdm import tqdm

from static_data.static_data import StaticData

PARCELL_FILE = 'CBIG/stable_projects/brain_parcellation/Schaefer2018_LocalGlobal/Parcellations/HCP/fslr32k/cifti/Schaefer2018_300Parcels_7Networks_order.dlabel.nii'


def load_results_from_s3(s3_client, bucket_name, s3_path):
    response = s3_client.get_object(Bucket=bucket_name, Key=s3_path)
    file_content = response['Body'].read()
    return pickle.loads(file_content)


def load_results(filepath):
    with open(filepath, 'rb') as f:
        return pickle.load(f)


def map_decoding_voxels_to_values(roi_array, roi_results, rest_between):
    scores_array = np.zeros(roi_array.shape)
    std_array = np.zeros(roi_array.shape)
    for array_pos, roi_index in tqdm(enumerate(roi_array)):
        if roi_index == 0:
            continue
        roi_name = _get_roi_name(roi_index=roi_index)
        if rest_between:
            mean_roi_score = roi_results[roi_name]["13-18"]["avg"]
            std_roi_score = roi_results[roi_name]["13-18"]["std"]
        else:
            mean_roi_score = roi_results[roi_name]["avg"]
            std_roi_score = roi_results[roi_name]["std"]

        scores_array[array_pos] = mean_roi_score
        std_array[array_pos] = std_roi_score

    return scores_array, std_array


def _get_roi_name(roi_index: int):
    roi_name_mapping = pd.read_csv("static_data/Schaefer2018_300Parcels_7Networks_order_FSLMNI152_1mm.Centroid_RAS.csv")
    roi_name = roi_name_mapping[roi_name_mapping["ROI Label"] == roi_index]["ROI Name"].values[0]
    roi_name = roi_name.replace("7Networks_", "")
    return roi_name


def main_decoding(rest_between: bool = False):
    results_pkl = 'results/svc_all_rois_17_groups_10_sub_in_group_rest_between_data_5TR_window_activations_results.pkl'
    if not rest_between:
        results_pkl = 'results/svc_all_rois_17_groups_10_sub_in_group_movie_data_last_5TR_activations_results.pkl'
    bucket_name = 'erezsimony'
    s3_client = boto3.client('s3')

    if not os.path.exists(results_pkl):
        s3_path = f"Results/{os.path.basename(results_pkl)}"
        try:
            roi_results = load_results_from_s3(s3_client, bucket_name, s3_path)
            print(f"Loaded {results_pkl} from S3")
        except s3_client.exceptions.NoSuchKey:
            raise FileNotFoundError(f"Result file {results_pkl} not found locally or in S3.")
    else:
        roi_results = load_results(results_pkl)

    left_surf, right_surf, left_data, right_data = load_surfaces_and_parcellation(
        roi_results,
        mapping_func=map_decoding_voxels_to_values,
        name="{score}_decoding_results" + ("_last_window_rest_between" if rest_between else "_last_movie_window"),
        rest_between=rest_between
    )
    plot_brain_surfaces(left_surf, right_surf, left_data, right_data)


def load_voxel_to_roi_mapping(csv_file):
    return pd.read_csv(csv_file)


def save_voxels_dscalar(parcel_template, array: np.array, name: str):
    vector = array[np.newaxis, :]  # Add an extra dimension, resulting in shape (1, 91282)
    new_dscalar = nib.Cifti2Image(vector, header=parcel_template.header, nifti_header=parcel_template.nifti_header)
    output_file = f"{name or 'output'}.dscalar.nii"
    nib.save(new_dscalar, output_file)
    print(f"Saved vector as .dscalar.nii file: {output_file}")


def load_surfaces_and_parcellation(roi_results, mapping_func: Callable, name: str, **kwargs):
    parcell_template = nib.load(PARCELL_FILE)
    data = parcell_template.get_fdata().squeeze()
    mean_arr, std_arr = mapping_func(data, roi_results, **kwargs)
    save_voxels_dscalar(parcell_template, mean_arr, name=name.format(score="mean"))
    save_voxels_dscalar(parcell_template, mean_arr, name=name.format(score="std"))
    surfaces = hcp.load_surfaces()
    left_surf = surfaces['inflated_left']
    right_surf = surfaces['inflated_right']
    n_left = len(left_surf[0])
    left_data = mean_arr[:n_left]
    right_data = mean_arr[n_left:]
    return left_surf, right_surf, left_data, right_data


def plot_brain_surfaces(left_surf, right_surf, left_data, right_data):
    fig_left = plt.figure(figsize=(8, 6))
    ax_left = fig_left.add_subplot(111, projection='3d')
    plot_surf_stat_map(
        surf_mesh=left_surf,
        stat_map=left_data,
        hemi='left',
        colorbar=True,
        title="Left Hemisphere (Schaefer 300 Parcellation)",
        cmap="cold_hot",
        axes=ax_left,
        threshold=None
    )
    fig_right = plt.figure(figsize=(8, 6))
    ax_right = fig_right.add_subplot(111, projection='3d')
    plot_surf_stat_map(
        surf_mesh=right_surf,
        stat_map=right_data,
        hemi='right',
        colorbar=True,
        title="Right Hemisphere (Schaefer 300 Parcellation)",
        cmap='cold_hot',
        axes=ax_right,
        threshold=None
    )
    plt.show()


def map_negative_correlation_voxels_to_values(roi_array, roi_results):
    scores_array = np.zeros(roi_array.shape)
    std_array = np.zeros(roi_array.shape)
    for array_pos, roi_index in tqdm(enumerate(roi_array)):
        if roi_index == 0:
            continue

        roi_name = _get_roi_name(roi_index=roi_index)
        roi_score = roi_results[roi_name]
        roi_mean_last_window = roi_score["mean"][-1]
        roi_std_last_window = roi_score["std"][-1]

        scores_array[array_pos] = roi_mean_last_window
        std_array[array_pos] = roi_std_last_window
    return scores_array, std_array


def main_negative_correlation(resting_state: bool):
    results_dir = "results/negative_correlation_resting_state"
    bucket_name = 'erezsimony'
    s3_client = boto3.client('s3')
    StaticData.inhabit_class_members()

    roi_results = {}
    for roi in tqdm(StaticData.ROI_NAMES):

        s3_path = f"Results/negative_correlation{'_resting_state' if resting_state else ''}/{roi}.pkl"
        try:
            # Load the file from S3 directly into memory
            roi_results[roi] = load_results_from_s3(s3_client, bucket_name, s3_path)
        except s3_client.exceptions.NoSuchKey:
            raise FileNotFoundError(f"Result file for {roi} not found in S3.")

    left_surf, right_surf, left_data, right_data = load_surfaces_and_parcellation(
        roi_results,
        mapping_func=map_negative_correlation_voxels_to_values,
        name="{score}_negative_correlation" + ("_resting_state" if resting_state else "")
    )
    plot_brain_surfaces(left_surf, right_surf, left_data, right_data)


if __name__ == '__main__':
    main_decoding()
    main_decoding(rest_between=True)
    main_negative_correlation(resting_state=True)
