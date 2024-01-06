from static_data.static_data import StaticData


class Utils:
    StaticData().inhabit_class_members()

    roi_list = StaticData.ROI_NAMES
    subject_list = StaticData.SUBJECTS
    movie_scan_mapping = StaticData.SCAN_MAPPING


def generate_windows_pair(k: int, n: int):
    for i in range(n - k + 1):
        if i + k <= n:
            yield i, i + k
