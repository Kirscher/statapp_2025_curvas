"""
Test the S3Singleton class with the updated has_changed method.

This script demonstrates how to use the S3Singleton class to check if a file has changed
based on checksums. It can be used to verify that the implementation works as expected.

Usage:
    python -m statapp.tests.s3.test_s3singleton <local_file_path> <remote_path>

    If no arguments are provided, it will use a sample file from the training_set directory.
"""

import os
import sys
import time
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Import the S3Singleton class
from statapp.S3Singleton import S3Singleton


def test_has_changed(local_path, remote_path):
    """
    Test the has_changed method of the S3Singleton class.

    Args:
        local_path (str): Path to the local file
        remote_path (str): Path in S3 where the file is stored (bucket/key format)
    """
    print(f"Testing S3Singleton.has_changed method:")
    print(f"Local file: {local_path}")
    print(f"Remote path: {remote_path}")

    # Check if local file exists
    if not os.path.exists(local_path):
        print(f"Error: Local file does not exist: {local_path}")
        return

    try:
        # Create an instance of S3Singleton
        s3 = S3Singleton()

        # Test the has_changed method
        start_time = time.time()
        has_changed = s3.has_changed(local_path, remote_path)
        end_time = time.time()

        # Print the result
        print(f"\nResult: File has {'changed' if has_changed else 'not changed'}")
        print(f"Time taken: {end_time - start_time:.4f} seconds")

        # Print file size for reference
        size = os.path.getsize(local_path)
        print(f"File size: {size} bytes")

        # For debugging, also print the MD5 hash of the local file
        local_md5 = s3._calculate_local_md5(local_path)
        print(f"Local file MD5: {local_md5}")

        # Try to get the ETag from S3 directly for comparison
        bucket, key = s3._parse_remote_path(remote_path, local_path)
        try:
            # Get object metadata
            response = s3._s3_client.head_object(Bucket=bucket, Key=key)
            etag = response.get("ETag", "").strip('"')

            # Handle multipart upload ETag
            if "-" in etag:
                print(f"Detected multipart upload ETag: {etag}")
                etag_parts = etag.split("-")
                etag_hash = etag_parts[0]
                print(f"ETag hash part: {etag_hash}")
                print(f"ETag matches local MD5: {etag_hash.lower() == local_md5.lower()}")
            else:
                print(f"S3 ETag: {etag}")
                print(f"ETag matches local MD5: {etag.lower() == local_md5.lower()}")
        except Exception as e:
            print(f"Error getting S3 object metadata: {str(e)}")

    except Exception as e:
        print(f"Error testing has_changed: {str(e)}")


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

    # Test the has_changed method
    test_has_changed(local_path, remote_path)


if __name__ == "__main__":
    main()