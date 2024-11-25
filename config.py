import os

# REPOSITORY PATH
ROOT_PATH = os.path.abspath(os.path.curdir)
DATASETS_PATH = os.path.join(ROOT_PATH, 'datasets')
RAW_DATASETS_PATH = os.path.join(DATASETS_PATH, 'raw_datasets')
PREPROCESSED_DATASETS_PATH = os.path.join(DATASETS_PATH, 'preprocessed_datasets')
EXPERIMENTS_RESULTS = os.path.join(ROOT_PATH, 'experiments', 'results')
FIGURES_PATH = os.path.join(ROOT_PATH, 'experiments', 'figures')

# DATA HELPERS
STATIC_DATA_PATH = os.path.join(ROOT_PATH, 'static_data')

# External source data path
DATA_DRIVE_E = os.path.join(r'/Volumes', 'My_Book', 'parcelled_data_niv')
SUBNET_DATA_DF = os.path.join(DATA_DRIVE_E, 'Schaefer2018_SUBNET_{mode}_DF')
SUBNET_AVG_N_SUBJECTS = os.path.join(DATA_DRIVE_E, 'Schaefer2018_SUBNET_AVG_N_SUBJECTS_{mode}',
                                     'AVG_{n_subjects}_SUBJECTS', 'GROUP_{group_i}')

# Raw data denormalized (without zscore)
VOXEL_DATA_DENORMALIZED = os.path.join(DATA_DRIVE_E, 'schaefer2018_VOXEL_denormalized')
VOXEL_DATA_DF_DENORMALIZED = os.path.join(DATA_DRIVE_E, 'schaefer2018_VOXEL_{mode}_DF_denormalized')
SUBNET_DATA_DF_DENORMALIZED = os.path.join(DATA_DRIVE_E, 'Schaefer2018_SUBNET_{mode}_DF_denormalized')
NETWORK_SUBNET_DATA_DF_DENORMALIZED = os.path.join(DATA_DRIVE_E, 'schaefer2018_NETWORKS_{mode}_DF_denormalized')

# S3
BUCKET_NAME = "erezsimony"
S3_SUBNET_DATA_DF_DENORMALIZED = 'Schaefer2018_SUBNET_{mode}_DF_denormalized'
S3_SUBNET_DATA_DF = 'Schaefer2018_SUBNET_{mode}_DF'

