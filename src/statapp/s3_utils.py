"""
S3 utilities module for the statapp application.

This module provides functions for interacting with S3 storage,
including uploading and downloading files and directories,
listing contents, and deleting objects.
"""

import os
import time
import tempfile
import shutil
from pathlib import Path
from typing import List, Dict, Any, Callable, Optional, Tuple, Union

from statapp.S3Singleton import S3Singleton


def list_data_directory() -> List[Dict[str, Any]]:
    """
    List contents of the data directory in the S3 bucket.

    Returns:
        List[Dict[str, Any]]: List of objects in the data directory
    """
    s3 = S3Singleton()
    return s3.list_data_directory()


def empty_data_directory() -> List[str]:
    """
    Delete all objects in the data directory of the S3 bucket.

    Returns:
        List[str]: List of deleted object keys
    """
    s3 = S3Singleton()
    return s3.empty_data_directory()


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
        
        # Create a temporary file for downloading
        temp_path = tempfile.mktemp()
        
        try:
            # Download the file with progress tracking
            s3.client.download_file(
                bucket, 
                key, 
                temp_path,
                Callback=callback,
                Config=s3._transfer_config
            )
            
            # Copy the file to the destination
            shutil.copy2(temp_path, local_path)
            
            # Verify the file was copied
            if not os.path.exists(local_path):
                raise FileNotFoundError(f"Failed to copy file to {local_path}")
                
            return True
        finally:
            # Clean up the temporary file
            if os.path.exists(temp_path):
                os.unlink(temp_path)
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
    uploaded_files = []
    s3 = S3Singleton()
    
    # Parse the remote path to get bucket and prefix
    bucket, prefix = parse_remote_path(remote_dir)
    
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
            
            # Upload the file
            try:
                s3.client.upload_file(
                    local_file_path, 
                    bucket, 
                    s3_key,
                    Callback=file_callback,
                    Config=s3._transfer_config
                )
                
                uploaded_files.append(f"{bucket}/{s3_key}")
            except Exception as e:
                import logging
                logging.error(f"Error uploading file {local_file_path} to {bucket}/{s3_key}: {str(e)}")
    
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
    downloaded_files = []
    s3 = S3Singleton()
    
    # Parse the remote path to get bucket and prefix
    bucket, prefix = parse_remote_path(remote_dir)
    
    # Ensure the local directory exists
    os.makedirs(local_dir, exist_ok=True)
    
    # List objects in the S3 directory
    paginator = s3.client.get_paginator('list_objects_v2')
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        if 'Contents' not in page:
            continue
            
        for obj in page['Contents']:
            key = obj['Key']
            
            # Skip if this is a directory marker
            if key.endswith('/'):
                continue
                
            # Calculate the relative path from the prefix
            rel_path = key[len(prefix):].lstrip('/')
            
            # Construct the local file path
            local_file_path = os.path.join(local_dir, rel_path)
            
            # Ensure the local directory exists
            os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
            
            # Create a callback wrapper that includes the filename if callback was provided
            file_callback = None
            if callback:
                filename = os.path.basename(key)
                def file_callback_fn(bytes_transferred):
                    callback(bytes_transferred, filename)
                file_callback = file_callback_fn
            
            # Download the file
            try:
                # Create a temporary file for downloading
                temp_path = tempfile.mktemp()
                
                try:
                    # Download the file with progress tracking
                    s3.client.download_file(
                        bucket, 
                        key, 
                        temp_path,
                        Callback=file_callback,
                        Config=s3._transfer_config
                    )
                    
                    # Copy the file to the destination
                    shutil.copy2(temp_path, local_file_path)
                    
                    # Verify the file was copied
                    if not os.path.exists(local_file_path):
                        raise FileNotFoundError(f"Failed to copy file to {local_file_path}")
                        
                    downloaded_files.append(local_file_path)
                finally:
                    # Clean up the temporary file
                    if os.path.exists(temp_path):
                        os.unlink(temp_path)
            except Exception as e:
                import logging
                logging.error(f"Error downloading file from {bucket}/{key} to {local_file_path}: {str(e)}")
    
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
    parts = remote_path.split('/', 1)
    bucket = parts[0]
    key = parts[1] if len(parts) > 1 else (os.path.basename(local_path) if local_path else "")
    return bucket, key