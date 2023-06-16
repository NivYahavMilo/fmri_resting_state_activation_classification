import os

# REPOSITORY PATH
ROOT_PATH = os.path.abspath(os.path.curdir)
DATASETS_PATH = os.path.join(ROOT_PATH, 'datasets')
RAW_DATASETS_PATH = os.path.join(DATASETS_PATH, 'raw_datasets')
PREPROCESSED_DATASETS_PATH = os.path.join(DATASETS_PATH, 'preprocessed_datasets')
EXPERIMENTS_RESULTS = os.path.join(ROOT_PATH, 'experiments', 'results')
FIGURES_PATH = os.path.join(ROOT_PATH, 'experiments', 'figures')


# External source data path
DATA_DRIVE_E = os.path.join(r'E:', 'parcelled_data_niv')
SUBNET_DATA_DF = os.path.join(DATA_DRIVE_E, 'Schaefer2018_SUBNET_{mode}_DF')
SUBNET_AVG_N_SUBJECTS = os.path.join(DATA_DRIVE_E, 'Schaefer2018_SUBNET_AVG_N_SUBJECTS_{mode}', 'AVG_{n_subjects}_SUBJECTS', 'GROUP_{group_i}')

