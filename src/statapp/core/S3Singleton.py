"""
S3 Singleton module for managing S3 connections using boto3.

This module provides a singleton class for accessing S3 storage,
ensuring only one connection is created throughout the application.
"""

import os
import boto3
import hashlib
from botocore.client import Config
from boto3.s3.transfer import TransferConfig
from typing import Optional, List, Callable, Any, Dict, Union, Tuple
from pathlib import Path


class S3Singleton:
    """
    Singleton class for S3 access using boto3.

    This class ensures only one boto3 S3 client instance is created
    and provides methods for interacting with the S3 storage.
    """

    _instance: Optional['S3Singleton'] = None
    _s3_client = None
    _s3_resource = None
    # Default transfer configuration for all uploads
    _transfer_config = TransferConfig(
        multipart_threshold=8 * 1024 * 1024,  # 8MB - use multipart upload for anything larger than this
        max_concurrency=10,                   # Reduced from 20 to 10 to prevent connection pool issues
        multipart_chunksize=8 * 1024 * 1024,  # 8MB per part - larger parts for better performance
        use_threads=True,                     # Use threads for faster uploads
        max_io_queue=50                       # Reduced from 100 to 50 to control memory usage
    )

    def __new__(cls) -> 'S3Singleton':
        """
        Create a new instance of S3Singleton if one doesn't exist.

        Returns:
            S3Singleton: The singleton instance
        """
        if cls._instance is None:
            cls._instance = super(S3Singleton, cls).__new__(cls)

            # Common connection parameters
            conn_params = {
                'endpoint_url': os.environ["S3_ENDPOINT"],
                'aws_access_key_id': os.environ["S3_ACCESS_KEY"],
                'aws_secret_access_key': os.environ["S3_SECRET_KEY"],
                'config': Config(
                    signature_version='s3v4',
                    max_pool_connections=25  # Reduced from 50 to 25 to prevent connection pool issues
                )
            }

            # Configure boto3 client and resource with the same parameters
            cls._instance._s3_client = boto3.client('s3', **conn_params)
            cls._instance._s3_resource = boto3.resource('s3', **conn_params)
        return cls._instance

    @property
    def client(self):
        """
        Get the boto3 S3 client instance.

        Returns:
            boto3.client.S3: The boto3 S3 client instance
        """
        return self._s3_client

    @property
    def resource(self):
        """
        Get the boto3 S3 resource instance.

        Returns:
            boto3.resource('s3'): The boto3 S3 resource instance
        """
        return self._s3_resource

    def list_data_directory(self) -> List[Dict[str, Any]]:
        """
        List contents of the data directory in the S3 bucket.

        Returns:
            List[Dict[str, Any]]: List of objects in the data directory
        """
        bucket = os.environ['S3_BUCKET']
        prefix = f"{os.environ['S3_DATA_DIR']}/"

        all_contents = []
        continuation_token = None

        while True:
            # Prepare parameters for list_objects_v2
            params = {
                'Bucket': bucket,
                'Prefix': prefix
            }

            # Add continuation token if we have one
            if continuation_token:
                params['ContinuationToken'] = continuation_token

            # Make the API call
            response = self._s3_client.list_objects_v2(**params)

            # Add contents to our result list
            if 'Contents' in response:
                all_contents.extend(response['Contents'])

            # Check if there are more objects to retrieve
            if not response.get('IsTruncated'):  # No more objects
                break

            # Get the continuation token for the next request
            continuation_token = response.get('NextContinuationToken')
            if not continuation_token:  # Safety check
                break

        return all_contents

    def list_artifacts_directory(self) -> List[Dict[str, Any]]:
        """
        List contents of the artifacts directory in the S3 bucket.

        Returns:
            List[Dict[str, Any]]: List of objects in the artifacts directory
        """
        bucket = os.environ['S3_BUCKET']
        prefix = f"{os.environ['S3_ARTIFACTS_DIR']}/"

        all_contents = []
        continuation_token = None

        while True:
            # Prepare parameters for list_objects_v2
            params = {
                'Bucket': bucket,
                'Prefix': prefix
            }

            # Add continuation token if we have one
            if continuation_token:
                params['ContinuationToken'] = continuation_token

            # Make the API call
            response = self._s3_client.list_objects_v2(**params)

            # Add contents to our result list
            if 'Contents' in response:
                all_contents.extend(response['Contents'])

            # Check if there are more objects to retrieve
            if not response.get('IsTruncated'):  # No more objects
                break

            # Get the continuation token for the next request
            continuation_token = response.get('NextContinuationToken')
            if not continuation_token:  # Safety check
                break

        return all_contents

    def list_output_directory(self) -> List[Dict[str, Any]]:
        """
        List contents of the output directory in the S3 bucket.

        Returns:
            List[Dict[str, Any]]: List of objects in the output directory
        """
        bucket = os.environ['S3_BUCKET']
        prefix = f"{os.environ['S3_OUTPUT_DIR']}/"

        all_contents = []
        continuation_token = None

        while True:
            # Prepare parameters for list_objects_v2
            params = {
                'Bucket': bucket,
                'Prefix': prefix
            }

            # Add continuation token if we have one
            if continuation_token:
                params['ContinuationToken'] = continuation_token

            # Make the API call
            response = self._s3_client.list_objects_v2(**params)

            # Add contents to our result list
            if 'Contents' in response:
                all_contents.extend(response['Contents'])

            # Check if there are more objects to retrieve
            if not response.get('IsTruncated'):  # No more objects
                break

            # Get the continuation token for the next request
            continuation_token = response.get('NextContinuationToken')
            if not continuation_token:  # Safety check
                break

        return all_contents

    def empty_data_directory(self) -> List[str]:
        """
        Delete all objects in the data directory of the S3 bucket.

        Returns:
            List[str]: List of deleted object keys
        """
        bucket = os.environ['S3_BUCKET']
        prefix = f"{os.environ['S3_DATA_DIR']}/"

        # List all objects in the data directory
        objects = self.list_data_directory()

        deleted_objects = []

        if objects:
            # Create a list of objects to delete
            delete_keys = [{'Key': obj['Key']} for obj in objects]

            # Delete the objects
            response = self._s3_client.delete_objects(
                Bucket=bucket,
                Delete={
                    'Objects': delete_keys,
                    'Quiet': False
                }
            )

            # Get the list of deleted objects
            if 'Deleted' in response:
                deleted_objects = [obj['Key'] for obj in response['Deleted']]

        return deleted_objects

    def empty_artifacts_directory(self) -> List[str]:
        """
        Delete all objects in the artifacts directory of the S3 bucket.

        Returns:
            List[str]: List of deleted object keys
        """
        bucket = os.environ['S3_BUCKET']
        prefix = f"{os.environ['S3_ARTIFACTS_DIR']}/"

        # List all objects in the artifacts directory
        objects = self.list_artifacts_directory()

        deleted_objects = []

        if objects:
            # Create a list of objects to delete
            delete_keys = [{'Key': obj['Key']} for obj in objects]

            # Delete the objects
            response = self._s3_client.delete_objects(
                Bucket=bucket,
                Delete={
                    'Objects': delete_keys,
                    'Quiet': False
                }
            )

            # Get the list of deleted objects
            if 'Deleted' in response:
                deleted_objects = [obj['Key'] for obj in response['Deleted']]

        return deleted_objects

    def upload_file(self, local_path: str, remote_path: str, callback: Callable[[int], None] = None) -> None:
        """
        Upload a file to S3 using boto3.

        Args:
            local_path (str): Path to the local file
            remote_path (str): Path in S3 where the file will be stored (bucket/key format)
            callback (callable, optional): Function to call with progress updates
                                          Should accept (bytes_transferred)
        """
        from statapp.utils import s3_utils
        s3_utils.upload_file(local_path, remote_path, callback)

    def download_file(self, bucket: str, key: str, local_path: str, callback: Callable[[int], None] = None) -> None:
        """
        Download a file from S3 using boto3.

        Args:
            bucket (str): S3 bucket name
            key (str): S3 object key
            local_path (str): Path where the file will be stored locally
            callback (callable, optional): Function to call with progress updates
                                          Should accept (bytes_transferred)
        """
        from statapp.utils import s3_utils
        s3_utils.download_file(f"{bucket}/{key}", local_path, callback)

    def upload_directory(self, local_dir: str, remote_dir: str, 
                        callback: Callable[[int, str], None] = None) -> List[str]:
        """
        Upload a directory to S3.

        Args:
            local_dir (str): Path to the local directory
            remote_dir (str): Path in S3 where the directory will be stored (bucket/prefix format)
            callback (callable, optional): Function to call with progress updates
                                          Should accept (bytes_transferred, filename)

        Returns:
            List[str]: List of uploaded files (bucket/key format)
        """
        from statapp.utils import s3_utils
        return s3_utils.upload_directory(local_dir, remote_dir, callback)

    def download_directory(self, remote_dir: str, local_dir: str,
                          callback: Callable[[int, str], None] = None) -> List[str]:
        """
        Download a directory from S3.

        Args:
            remote_dir (str): Path in S3 where the directory is stored (bucket/prefix format)
            local_dir (str): Path where the directory will be stored locally
            callback (callable, optional): Function to call with progress updates
                                          Should accept (bytes_transferred, filename)

        Returns:
            List[str]: List of downloaded files (local paths)
        """
        from statapp.utils import s3_utils
        return s3_utils.download_directory(remote_dir, local_dir, callback)
