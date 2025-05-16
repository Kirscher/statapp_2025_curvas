"""
S3 utilities module for the statapp application.

This module provides functions for interacting with S3 storage,
including uploading and downloading files and directories,
listing contents, and deleting objects.
"""

import os
import shutil
import tempfile
from typing import List, Dict, Any, Callable, Tuple

from statapp.core.S3Singleton import S3Singleton


def list_data_directory() -> List[Dict[str, Any]]:
    """
    List contents of the data directory in the S3 bucket.

    Returns:
        List[Dict[str, Any]]: List of objects in the data directory
    """
    s3 = S3Singleton()
    return s3.list_data_directory()


def list_artifacts_directory() -> List[Dict[str, Any]]:
    """
    List contents of the artifacts directory in the S3 bucket.

    Returns:
        List[Dict[str, Any]]: List of objects in the artifacts directory
    """
    s3 = S3Singleton()
    return s3.list_artifacts_directory()


def list_output_directory() -> List[Dict[str, Any]]:
    """
    List contents of the output directory in the S3 bucket.

    Returns:
        List[Dict[str, Any]]: List of objects in the output directory
    """
    s3 = S3Singleton()
    return s3.list_output_directory()


def list_metrics_directory() -> List[Dict[str, Any]]:
    """
    List contents of the metrics directory in the S3 bucket.

    Returns:
        List[Dict[str, Any]]: List of objects in the metrics directory
    """
    s3 = S3Singleton()
    return s3.list_metrics_directory()


def empty_data_directory() -> List[str]:
    """
    Delete all objects in the data directory of the S3 bucket.

    Returns:
        List[str]: List of deleted object keys
    """
    s3 = S3Singleton()
    return s3.empty_data_directory()


def empty_artifacts_directory() -> List[str]:
    """
    Delete all objects in the artifacts directory of the S3 bucket.

    Returns:
        List[str]: List of deleted object keys
    """
    s3 = S3Singleton()
    return s3.empty_artifacts_directory()


def get_file_size(bucket: str, key: str) -> int:
    """
    Get the size of a file in S3.

    Args:
        bucket (str): S3 bucket name
        key (str): S3 object key

    Returns:
        int: Size of the file in bytes, or 0 if the file doesn't exist
    """
    try:
        s3 = S3Singleton()
        response = s3.client.head_object(Bucket=bucket, Key=key)
        return response.get('ContentLength', 0)
    except Exception:
        return 0


def upload_file(local_path: str, remote_path: str, callback: Callable[[int], None] = None) -> bool:
    """
    Upload a file to S3.

    Args:
        local_path (str): Path to the local file
        remote_path (str): Path in S3 where the file will be stored (bucket/key format)
        callback (callable, optional): Function to call with progress updates
                                      Should accept (bytes_transferred)

    Returns:
        bool: True if the upload was successful, False otherwise
    """
    try:
        s3 = S3Singleton()
        bucket, key = parse_remote_path(remote_path, local_path)

        # Upload the file with progress tracking
        s3.client.upload_file(
            local_path, 
            bucket, 
            key,
            Callback=callback,
            Config=s3._transfer_config
        )
        return True
    except Exception as e:
        import logging
        logging.error(f"Error uploading file {local_path} to {remote_path}: {str(e)}")
        return False


def get_file_md5(file_path: str) -> str:
    """
    Calculate MD5 hash of a file.

    Args:
        file_path (str): Path to the file

    Returns:
        str: MD5 hash of the file
    """
    import hashlib

    md5_hash = hashlib.md5()
    with open(file_path, "rb") as f:
        # Read in chunks to handle large files
        for chunk in iter(lambda: f.read(4096), b""):
            md5_hash.update(chunk)
    return md5_hash.hexdigest()

def download_file(remote_path: str, local_path: str, callback: Callable[[int], None] = None) -> bool:
    """
    Download a file from S3.

    Args:
        remote_path (str): Path in S3 where the file is stored (bucket/key format)
        local_path (str): Path where the file will be stored locally
        callback (callable, optional): Function to call with progress updates
                                      Should accept (bytes_transferred)

    Returns:
        bool: True if the download was successful, False otherwise
    """
    try:
        s3 = S3Singleton()
        bucket, key = parse_remote_path(remote_path)

        # Ensure the destination directory exists
        os.makedirs(os.path.dirname(local_path), exist_ok=True)

        # Check if file exists locally and compare with S3 ETag
        if os.path.exists(local_path):
            try:
                # Get S3 object metadata
                response = s3.client.head_object(Bucket=bucket, Key=key)
                etag = response.get('ETag', '').strip('"')

                # Calculate MD5 of local file
                local_md5 = get_file_md5(local_path)

                # If ETags match, file is unchanged - skip download
                if etag and local_md5 == etag:
                    import logging
                    logging.info(f"File {local_path} is unchanged, skipping download")
                    return True
            except Exception as e:
                # If there's an error checking ETag, proceed with download
                import logging
                logging.warning(f"Error checking ETag for {remote_path}: {str(e)}")

        # Download directly to the destination file
        s3.client.download_file(
            bucket, 
            key, 
            local_path,
            Callback=callback,
            Config=s3._transfer_config
        )

        # Verify the file was downloaded
        if not os.path.exists(local_path):
            raise FileNotFoundError(f"Failed to download file to {local_path}")

        return True
    except Exception as e:
        import logging
        logging.error(f"Error downloading file from {remote_path} to {local_path}: {str(e)}")
        return False


def upload_directory(local_dir: str, remote_dir: str, 
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
    import concurrent.futures
    import logging

    uploaded_files = []
    s3 = S3Singleton()

    # Parse the remote path to get bucket and prefix
    bucket, prefix = parse_remote_path(remote_dir)

    # Function to upload a single file
    def upload_file_task(local_file_path, bucket, s3_key, file_callback=None):
        try:
            s3.client.upload_file(
                local_file_path, 
                bucket, 
                s3_key,
                Callback=file_callback,
                Config=s3._transfer_config
            )
            return f"{bucket}/{s3_key}"
        except Exception as e:
            logging.error(f"Error uploading file {local_file_path} to {bucket}/{s3_key}: {str(e)}")
            return None

    # Collect all files to upload
    files_to_upload = []
    for root, _, files in os.walk(local_dir):
        for file in files:
            local_file_path = os.path.join(root, file)
            # Calculate the relative path from the base directory
            rel_path = os.path.relpath(local_file_path, local_dir)
            # Construct the S3 key with the prefix and relative path
            s3_key = os.path.join(prefix, rel_path).replace('\\', '/')
            files_to_upload.append((local_file_path, file, s3_key))

    # Process files in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
        future_to_file = {}

        for local_file_path, file, s3_key in files_to_upload:
            # Create a callback wrapper that includes the filename if callback was provided
            file_callback = None
            if callback:
                def file_callback_fn(bytes_transferred):
                    callback(bytes_transferred, file)
                file_callback = file_callback_fn

            # Submit upload task to the executor
            future = executor.submit(
                upload_file_task,
                local_file_path,
                bucket,
                s3_key,
                file_callback
            )
            future_to_file[future] = local_file_path

        # Process results as they complete
        for future in concurrent.futures.as_completed(future_to_file):
            result = future.result()
            if result:
                uploaded_files.append(result)

    return uploaded_files


def download_directory(remote_dir: str, local_dir: str,
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
    import concurrent.futures
    import logging

    downloaded_files = []
    s3 = S3Singleton()

    # Parse the remote path to get bucket and prefix
    bucket, prefix = parse_remote_path(remote_dir)

    # Ensure the local directory exists
    os.makedirs(local_dir, exist_ok=True)

    # Function to download a single file
    def download_file_task(bucket, key, local_file_path, file_callback=None):
        try:
            # Ensure the local directory exists
            os.makedirs(os.path.dirname(local_file_path), exist_ok=True)

            # Check if file exists locally and compare with S3 ETag
            if os.path.exists(local_file_path):
                try:
                    # Get S3 object metadata
                    response = s3.client.head_object(Bucket=bucket, Key=key)
                    etag = response.get('ETag', '').strip('"')

                    # Calculate MD5 of local file
                    local_md5 = get_file_md5(local_file_path)

                    # If ETags match, file is unchanged - skip download
                    if etag and local_md5 == etag:
                        logging.info(f"File {local_file_path} is unchanged, skipping download")
                        return local_file_path
                except Exception as e:
                    # If there's an error checking ETag, proceed with download
                    logging.warning(f"Error checking ETag for {bucket}/{key}: {str(e)}")

            # Download the file with progress tracking directly to destination
            s3.client.download_file(
                bucket, 
                key, 
                local_file_path,
                Callback=file_callback,
                Config=s3._transfer_config
            )

            # Verify the file was downloaded
            if not os.path.exists(local_file_path):
                raise FileNotFoundError(f"Failed to download file to {local_file_path}")

            return local_file_path
        except Exception as e:
            logging.error(f"Error downloading file from {bucket}/{key} to {local_file_path}: {str(e)}")
            return None

    # Collect all objects to download
    objects = []
    paginator = s3.client.get_paginator('list_objects_v2')
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        if 'Contents' in page:
            objects.extend(page['Contents'])

    # Reduce the number of concurrent downloads to avoid connection pool issues
    max_workers = 10  # Reduced from 20 to 10

    # Process files in parallel with a reduced number of workers
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_file = {}
        batch_size = 10  # Process files in smaller batches

        for i in range(0, len(objects), batch_size):
            batch = objects[i:i+batch_size]

            # Process this batch
            for obj in batch:
                key = obj['Key']

                # Skip if this is a directory marker
                if key.endswith('/'):
                    continue

                # Calculate the relative path from the prefix
                rel_path = key[len(prefix):].lstrip('/')

                # Construct the local file path
                local_file_path = os.path.join(local_dir, rel_path)

                # Create a callback wrapper that includes the filename if callback was provided
                file_callback = None
                if callback:
                    filename = os.path.basename(key)
                    def file_callback_fn(bytes_transferred):
                        callback(bytes_transferred, filename)
                    file_callback = file_callback_fn

                # Submit download task to the executor
                future = executor.submit(
                    download_file_task, 
                    bucket, 
                    key, 
                    local_file_path, 
                    file_callback
                )
                future_to_file[future] = key

            # Process results for this batch as they complete
            for future in concurrent.futures.as_completed(list(future_to_file.keys())):
                result = future.result()
                if result:
                    downloaded_files.append(result)
                # Remove the future from the dictionary to free up resources
                del future_to_file[future]

    return downloaded_files


def parse_remote_path(remote_path: str, local_path: str = None) -> Tuple[str, str]:
    """
    Parse a remote path into bucket and key components.

    Args:
        remote_path (str): Path in S3 (bucket/key format)
        local_path (str, optional): Local path to use for the key if not specified in remote_path

    Returns:
        Tuple[str, str]: Bucket and key
    """
    # Check if the remote path starts with the artifacts, data, output, or metrics directory
    if remote_path.startswith(os.environ['S3_ARTIFACTS_DIR']) or remote_path.startswith(os.environ['S3_DATA_DIR']) or remote_path.startswith(os.environ['S3_OUTPUT_DIR']) or remote_path.startswith(os.environ['S3_METRICS_DIR']):
        # Use the correct bucket name from the environment variable
        bucket = os.environ['S3_BUCKET']
        key = remote_path
    else:
        # Original behavior for other paths
        parts = remote_path.split('/', 1)
        bucket = parts[0]
        key = parts[1] if len(parts) > 1 else (os.path.basename(local_path) if local_path else "")
    return bucket, key
