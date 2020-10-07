import datetime
import os

import ee
import yaml
import tensorflow as tf
from google.cloud import storage

from DataPreparation.DatasetPrepareService import DatasetPrepareService

with open("DataPreparation/config/configuration.yml", "r", encoding="utf8") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
ee.Initialize()

def download_blob(bucket_name, prefix, destination_file_name):
    """Downloads a blob from the bucket."""
    # bucket_name = "your-bucket-name"
    # source_blob_name = "storage-object-name"
    # destination_file_name = "local/path/to/file"

    storage_client = storage.Client()

    bucket = storage_client.bucket(bucket_name)
    blobs = bucket.list_blobs(prefix=prefix)
    for blob in blobs:
        filename = blob.name.replace('/', '_')
        blob.download_to_filename(destination_file_name + filename)

    print(
        "Blob {} downloaded to {}.".format(
            filename, destination_file_name
        )
    )

if __name__ == '__main__':
    satellites = ['GOES']
    locations = ['August_complex']
    for location in locations:
        for satellite in satellites:
            blob_name = location + satellite + '/'
            destination_name = 'data/' + location + satellite + '/'
            dir_name = os.path.dirname(destination_name)
            if not os.path.exists(dir_name):
                os.makedirs(dir_name)
            download_blob(config.get('output_bucket'), blob_name, destination_name)