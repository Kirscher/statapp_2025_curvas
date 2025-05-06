"""
S3 utilities module for the statapp application.

This module provides functions for interacting with S3 storage,
including uploading files and directories, listing contents,
and deleting objects.
"""

from typing import List, Callable, Any, Dict

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

def upload_file(local_path: str, remote_path: str, callback: Callable[[int], None] = None) -> None:
    """
    Upload a file to S3 using boto3.

    Args:
        local_path (str): Path to the local file
        remote_path (str): Path in S3 where the file will be stored
        callback (callable, optional): Function to call with progress updates
                                      Should accept (bytes_transferred)
    """
    s3 = S3Singleton()
    s3.upload_file(local_path, remote_path, callback=callback)

def upload_directory(local_dir: str, remote_dir: str, 
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
    s3 = S3Singleton()
    return s3.upload_directory(local_dir, remote_dir, callback=callback)