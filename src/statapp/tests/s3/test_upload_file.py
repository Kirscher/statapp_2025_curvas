"""
Test uploading a file to S3 and verify that the local and remote files are identical.

This script uploads a file to S3 using the S3Singleton class and then verifies
that the local file and the S3 file have the same hash.

Usage:
    python -m statapp.tests.s3.test_upload_file <local_file_path> <remote_path>

    If no arguments are provided, it will use a sample file from the training_set directory.
"""

import os
import sys
import time
import hashlib
from pathlib import Path
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


def upload_and_verify(local_path, remote_path):
    """
    Upload a file to S3 and verify that the local and remote files are identical.
    
    Args:
        local_path (str): Path to the local file
        remote_path (str): Path in S3 where the file will be stored (bucket/key format)
    """
    print(f"Uploading file to S3 and verifying hashes:")
    print(f"Local file: {local_path}")
    print(f"Remote path: {remote_path}")
    
    # Check if local file exists
    if not os.path.exists(local_path):
        print(f"Error: Local file does not exist: {local_path}")
        return
    
    try:
        # Create an instance of S3Singleton
        s3 = S3Singleton()
        
        # Calculate local MD5 hash before upload
        local_md5_before = calculate_md5(local_path)
        print(f"Local file MD5 before upload: {local_md5_before}")
        
        # Upload the file
        print(f"Uploading file to S3...")
        start_time = time.time()
        s3.upload_file(local_path, remote_path)
        end_time = time.time()
        print(f"Upload completed in {end_time - start_time:.4f} seconds")
        
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
        
        # Calculate local MD5 hash after upload (should be the same as before)
        local_md5_after = calculate_md5(local_path)
        print(f"Local file MD5 after upload: {local_md5_after}")
        
        # Compare hashes
        if local_md5_after.lower() == etag_hash.lower():
            print(f"\nSuccess! Local file and S3 object have the same hash.")
        else:
            print(f"\nWarning: Local file and S3 object have different hashes.")
            print(f"Local MD5: {local_md5_after}")
            print(f"S3 ETag: {etag_hash}")
        
        # Check if the local file has changed according to S3Singleton
        has_changed = s3.has_changed(local_path, remote_path)
        print(f"\nS3Singleton.has_changed reports: File has {'changed' if has_changed else 'not changed'}")
        
    except Exception as e:
        print(f"Error during upload and verification: {str(e)}")


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
    
    # Upload the file and verify hashes
    upload_and_verify(local_path, remote_path)


if __name__ == "__main__":
    main()