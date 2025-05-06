"""
S3 Singleton module for managing S3 connections using boto3.

This module provides a singleton class for accessing S3 storage,
ensuring only one connection is created throughout the application.
"""

import os
import boto3
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
        multipart_threshold=1024 * 25,  # 25KB - use multipart upload for anything larger than this
        max_concurrency=10,             # Number of threads for concurrent uploads
        multipart_chunksize=1024 * 25,  # 25KB per part - smaller parts means more frequent callbacks
        use_threads=True                # Use threads for faster uploads
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
                'config': Config(signature_version='s3v4')
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

        response = self._s3_client.list_objects_v2(
            Bucket=bucket,
            Prefix=prefix
        )

        if 'Contents' in response:
            return response['Contents']
        return []

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

    def _parse_remote_path(self, remote_path: str, local_path: str = None) -> Tuple[str, str]:
        """
        Parse a remote path into bucket and key components.

        Args:
            remote_path (str): Path in S3 (bucket/key format)
            local_path (str, optional): Local path to use for the key if not specified in remote_path

        Returns:
            Tuple[str, str]: Bucket and key
        """
        parts = remote_path.split('/', 1)
        bucket = parts[0]
        key = parts[1] if len(parts) > 1 else (os.path.basename(local_path) if local_path else "")
        return bucket, key

    def has_changed(self, local_path: str, remote_path: str) -> bool:
        """
        Tests if a file has been changed based on the timestaps.

        Args:
            local_path (str): relative path of the local file
            remote_path (str): path of the remote file

        Returns:
            bool: indicates if the file has changed
        """
        bucket, key = self._parse_remote_path(remote_path, local_path)
        remote = self._s3_client.get_object(Bucket=bucket, Key=key)
        print(remote['LastModified'])
        return int(remote["LastModified"].strftime('%s')) != int(os.path.getmtime(local_path))


    def upload_file(self, local_path: str, remote_path: str, callback: Callable[[int], None] = None) -> None:
        """
        Upload a file to S3 using boto3.

        Args:
            local_path (str): Path to the local file
            remote_path (str): Path in S3 where the file will be stored
            callback (callable, optional): Function to call with progress updates
                                          Should accept (bytes_transferred)
        """
        if (not self.has_changed(local_path, remote_path)):
            return

        bucket, key = self._parse_remote_path(remote_path, local_path)

        # Upload the file with progress tracking
        self._s3_client.upload_file(
            local_path, 
            bucket, 
            key,
            Callback=callback,
            Config=self._transfer_config
        )

    def upload_directory(self, local_dir: str, remote_dir: str, 
                        callback: Callable[[int, str], None] = None) -> List[str]:
        """
        Upload a directory to S3.

        Args:
            local_dir (str): Path to the local directory
            remote_dir (str): Path in S3 where the directory will be stored
            callback (callable, optional): Function to call with progress updates
                                          Should accept (bytes_transferred, filename)

        Returns:
            List[str]: List of uploaded files
        """
        uploaded_files = []
        local_path = Path(local_dir)

        # Parse the remote path to get bucket and prefix
        bucket, prefix = self._parse_remote_path(remote_dir)

        # Walk through the directory and upload each file
        for root, _, files in os.walk(local_dir):
            for file in files:
                local_file_path = os.path.join(root, file)

                # Calculate the relative path from the base directory
                rel_path = os.path.relpath(local_file_path, local_dir)

                # Construct the S3 key with the prefix and relative path
                s3_key = os.path.join(prefix, rel_path).replace('\\', '/')

                # Create a callback wrapper that includes the filename if callback was provided
                file_callback = None
                if callback:
                    def file_callback_fn(bytes_transferred):
                        callback(bytes_transferred, file)
                    file_callback = file_callback_fn

                self._s3_client.upload_file(
                    local_file_path, 
                    bucket, 
                    s3_key,
                    Callback=file_callback,
                    Config=self._transfer_config
                )

                uploaded_files.append(f"{bucket}/{s3_key}")

        return uploaded_files
