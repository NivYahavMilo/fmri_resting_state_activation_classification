import boto3
import pandas as pd


def fetch_object_from_s3(s3_client: boto3.client, bucket_name: str, object_name: str):
    try:
        response = s3_client.get_object(Bucket=bucket_name, Key=object_name)
        s3_object = pd.read_pickle(response['Body'])
        print(f"Fetched {object_name} successfully from S3.")
        return s3_object

    except s3_client.exceptions.ClientError as e:
        if e.response['Error']['Code'] == "NoSuchKey":
            print(f"{object_name} not found in S3. Proceeding with processing.")
