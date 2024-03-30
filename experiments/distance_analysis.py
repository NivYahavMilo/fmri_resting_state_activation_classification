import os.path

import pandas as pd

import config


def compare_groups_rest_pattern():
    group_data_path = os.path.join(config.RAW_DATASETS_PATH,
                                   'RH_DorsAttn_Post_2',
                                   'distances_group_subjects_RH_DorsAttn_Post_2_roi_(13, 18)_rest_window.pkl')
    _group_data = pd.read_pickle(group_data_path)
    movie_template_group = _group_data[1]['task']

    for rest_group in range(2, 7):
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
        rest_data = _group_data[rest_group]['rest']
        precessed_features = features.drop(features[features == 1].index)
        return precessed_features

        print()


if __name__ == '__main__':
    compare_groups_rest_pattern()
