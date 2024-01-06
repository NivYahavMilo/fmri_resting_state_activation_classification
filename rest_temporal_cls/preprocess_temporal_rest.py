from tqdm import tqdm

from rest_temporal_cls.sequence_normalizer import get_normalized_data
from rest_temporal_cls.utils import generate_windows_pair


def get_temporal_rest_window_activations(roi: str):
    normalized_data = get_normalized_data(roi=roi)
    subjects_data = {}
    subjects = normalized_data['Subject'].unique()
    print("Preprocessing Data")
    for sub in tqdm(subjects):
        preprocess_data = {}
        for rest_i in range(1, 15):
            rest_seq = normalized_data[
                (normalized_data['y'] == rest_i)
                & (normalized_data['timepoint'].isin(range(0, 19)))
                & (normalized_data['Subject'] == sub)
                ]
            # Assuming df is your DataFrame
            half_point = len(rest_seq) // 2
            rest_seq = rest_seq.tail(len(rest_seq) - half_point)

            preprocess_movie = {}
            for w_i, (window_start, window_end) in enumerate(generate_windows_pair(k=5, n=19)):
                window_seq = rest_seq[rest_seq['timepoint'].isin(range(window_start, window_end))]
                window_seq = window_seq.drop(['y', 'timepoint', 'Subject'], axis=1)
                preprocess_movie[f'{window_start}-{window_end}'] = window_seq.values.flatten()

            preprocess_data[rest_i] = preprocess_movie

        subjects_data[sub] = preprocess_data

    return subjects_data
