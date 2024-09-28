from movie_temporal_cls.train_temporal_sequence import evaluate_movie_windows
from rest_temporal_cls.train_temporal_sequence import evaluate_rest_windows
from static_data.static_data import StaticData


def main():
    #roi_list = ["RH_Default_PFCdPFCm_7"]
    StaticData.inhabit_class_members()
    roi_list = StaticData.ROI_NAMES

    window_size = 5
    groups = 17
    subject_in_group = 10
    evaluate_movie_windows(
        rois=roi_list,
        distances=False,
        checkpoint=True,
        validation="llo",
        group_average=True,
        k_subjects_in_group=subject_in_group,
        k_split=groups,
        k_window_size=window_size,
        window_preprocess_method="mean",
        output_name=f"all_rois_{groups}_groups_{subject_in_group}_sub_in_group_movie_data_last_{window_size}TR_activations_results.pkl"

    )

    evaluate_rest_windows(
        distances=False,
        rois=roi_list,
        checkpoint=True,
        validation="llo",
        group_average=True,
        group_mean_correlation=False,
        k_subjects_in_group=subject_in_group,
        k_window_size=window_size,
        k_split=groups,
        n_timepoints=18,
        window_preprocess_method="mean",  # mean or flattening
        output_name=f'all_rois_{groups}_groups_{subject_in_group}_sub_in_group_rest_between_data_{window_size}TR_window_activations_results.pkl'

    )


if __name__ == '__main__':
    main()
