import os

import boto3
import tqdm
from botocore.exceptions import NoCredentialsError, ClientError

s3_client = boto3.client('s3')


def file_exists(bucket_name, s3_path):
    """
    Check if a file already exists in an S3 bucket.

    :param bucket_name: Name of the S3 bucket.
    :param s3_path: Path (key) to the file in S3.
    :return: True if the file exists, False otherwise.
    """
    try:
        s3_client.head_object(Bucket=bucket_name, Key=s3_path)
        return True  # File exists
    except ClientError as e:
        if e.response['Error']['Code'] == '404':
            return False  # File does not exist
        else:
            # Other errors, such as permission issues
            raise


def upload_folder_to_s3(local_folder, bucket_name, s3_folder=None):
    """
    Recursively upload a directory and its contents to an S3 bucket, preserving the directory structure.

    :param local_folder: Path to the local folder you want to upload
    :param bucket_name: S3 bucket name
    :param s3_folder: Optional, S3 folder name (to upload into a specific directory within the bucket)
    """
    global s3_client

    # Walk through each file and subfolder in the local folder
    for root, dirs, files in os.walk(local_folder):
        for subject_folder in tqdm.tqdm(dirs):
            local_path = os.path.join(root, subject_folder)

            # Construct the S3 path, including any subdirectories
            files_to_upload = [file for file in os.listdir(local_path) if not file.startswith('._')]
            try:
                for file in tqdm.tqdm(files_to_upload):
                    file_path = os.path.join(local_path, file)
                    s3_path = file_path if s3_folder is None else os.path.join(s3_folder, subject_folder, file)
                    if file_exists(bucket_name, s3_path):
                        continue
                    # Upload the file to S3
                    s3_client.upload_file(file_path, bucket_name, s3_path)
                    # print(f"Uploaded: {local_path} to s3://{bucket_name}/{s3_path}")
            except FileNotFoundError:
                print(f"File not found: {local_path}")
            except NoCredentialsError:
                print("Credentials not available")


if __name__ == '__main__':

    bucket_name = "erezsimony"  # Replace with your S3 bucket name
    # s3_folder = ""  # Replace with your target folder in S3, or None to upload to root

    dirs_to_upload = [
        "Schaefer2018_SUBNET_TASK_DF_denormalized"
        # "Schaefer2018_SUBNET_FIRST_REST_SECTION_DF_denormalized",
        # "Schaefer2018_SUBNET_REST_DF_denormalized",
        # "Schaefer2018_SUBNET_RESTING_STATE_REST_DF_denormalized",
        # "Schaefer2018_SUBNET_RESTING_STATE_TASK_DF_denormalized",
        # "Schaefer2018_SUBNET_TASK_DF"
    ]

    for directory in dirs_to_upload:
        print(f"Uploading {directory}")
        DATA_DRIVE_E = os.path.join(r'/Volumes', 'My_Book', 'parcelled_data_niv')
        upload_folder_to_s3(local_folder=os.path.join(DATA_DRIVE_E, directory), bucket_name=bucket_name,
                            s3_folder=directory)
        print(f"Finished uploading {directory}")
