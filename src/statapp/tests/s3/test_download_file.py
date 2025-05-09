"""
Test downloading a file from S3 and verify that it matches the local file.

This script downloads a file from S3 and calculates its MD5 hash to compare
with the hash of the local file. This helps determine if the file content
is actually different or if it's just the way S3 calculates the ETag.

Usage:
    python -m statapp.tests.s3.test_download_file <local_file_path> <remote_path>

    If no arguments are provided, it will use a sample file from the training_set directory.
"""

import hashlib
import os
import sys
import tempfile
import time

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Import the S3Singleton class
from statapp.S3Singleton import S3Singleton


def calculate_md5(file_path):
    """
    Calculate MD5 hash for a file.
    
    Args:
        file_path (str): Path to the file
        
    Returns:
        str: MD5 hash of the file content
    """
    md5_hash = hashlib.md5()
    try:
        with open(file_path, "rb") as f:
            # Read the file in chunks to handle large files efficiently
            for chunk in iter(lambda: f.read(4096), b""):
                md5_hash.update(chunk)
        return md5_hash.hexdigest()
    except Exception as e:
        print(f"Error calculating hash for {file_path}: {str(e)}")
        return None


def download_and_verify(local_path, remote_path):
    """
    Download a file from S3 and verify that it matches the local file.
    
    Args:
        local_path (str): Path to the local file
        remote_path (str): Path in S3 where the file is stored (bucket/key format)
    """
    print(f"Downloading file from S3 and verifying hashes:")
    print(f"Local file: {local_path}")
    print(f"Remote path: {remote_path}")
    
    # Check if local file exists
    if not os.path.exists(local_path):
        print(f"Error: Local file does not exist: {local_path}")
        return
    
    try:
        # Create an instance of S3Singleton
        s3 = S3Singleton()
        
        # Calculate local MD5 hash
        local_md5 = calculate_md5(local_path)
        print(f"Local file MD5: {local_md5}")
        
        # Parse the remote path to get bucket and key
        bucket, key = s3._parse_remote_path(remote_path, local_path)
        
        # Get the ETag from S3
        response = s3._s3_client.head_object(Bucket=bucket, Key=key)
        etag = response.get("ETag", "").strip('"')
        
        # Handle multipart upload ETag
        if "-" in etag:
            print(f"Detected multipart upload ETag: {etag}")
            etag_parts = etag.split("-")
            etag_hash = etag_parts[0]
            print(f"ETag hash part: {etag_hash}")
        else:
            etag_hash = etag
            print(f"S3 ETag: {etag_hash}")
        
        # Create a temporary file to download the S3 object
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_path = temp_file.name
        
        # Download the file from S3
        print(f"Downloading file from S3...")
        start_time = time.time()
        s3._s3_client.download_file(bucket, key, temp_path)
        end_time = time.time()
        print(f"Download completed in {end_time - start_time:.4f} seconds")
        
        # Calculate MD5 hash of the downloaded file
        downloaded_md5 = calculate_md5(temp_path)
        print(f"Downloaded file MD5: {downloaded_md5}")
        
        # Compare hashes
        if local_md5.lower() == downloaded_md5.lower():
            print(f"\nSuccess! Local file and downloaded S3 object have the same hash.")
        else:
            print(f"\nWarning: Local file and downloaded S3 object have different hashes.")
            print(f"Local MD5: {local_md5}")
            print(f"Downloaded MD5: {downloaded_md5}")
        
        # Compare downloaded file hash with S3 ETag
        if downloaded_md5.lower() == etag_hash.lower():
            print(f"\nSuccess! Downloaded file hash matches S3 ETag.")
        else:
            print(f"\nWarning: Downloaded file hash does not match S3 ETag.")
            print(f"Downloaded MD5: {downloaded_md5}")
            print(f"S3 ETag: {etag_hash}")
        
        # Clean up the temporary file
        os.unlink(temp_path)
        
    except Exception as e:
        print(f"Error during download and verification: {str(e)}")


def main():
    """Main function to handle command line arguments and execute the test."""
    # Check if arguments were provided
    if len(sys.argv) == 3:
        local_path = sys.argv[1]
        remote_path = sys.argv[2]
    else:
        # Use default values from the training_set directory
        training_dir = r"C:\Users\Concordance\Downloads\training_set\training_set"
        # Find the first .nii.gz file in the directory
        local_path = None
        for root, _, files in os.walk(training_dir):
            for file in files:
                if file.endswith(".nii.gz"):
                    local_path = os.path.join(root, file)
                    break
            if local_path:
                break
        
        if not local_path:
            print(f"Error: No .nii.gz files found in {training_dir}")
            return
        
        # Use environment variables for S3 bucket and construct key from local path
        bucket = os.environ.get("S3_BUCKET", "projet-statapp-segmedic")
        # Extract the relative path from the training_set directory
        rel_path = os.path.relpath(local_path, training_dir)
        # Construct the remote path (bucket/data/relative_path)
        remote_path = f"{bucket}/data/{rel_path.replace('\\', '/')}"
        
        print(f"Using default values:")
        print(f"Local path: {local_path}")
        print(f"Remote path: {remote_path}")
    
    # Download the file and verify hashes
    download_and_verify(local_path, remote_path)


if __name__ == "__main__":
    main()