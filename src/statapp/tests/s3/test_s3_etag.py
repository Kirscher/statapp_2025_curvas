"""
Test S3 ETag comparison with local file MD5 hash.

This script demonstrates how to extract the ETag from an S3 object and compare it
with the MD5 hash of a local file, similar to what the S3Singleton.has_changed method does.

Usage:
    python -m statapp.tests.s3.test_s3_etag <local_file_path> <s3_bucket> <s3_key>

    If no arguments are provided, it will use a sample file from the training_set directory.
"""

import hashlib
import os
import sys
import boto3
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


def calculate_local_md5(file_path):
    """
    Calculate MD5 hash for a local file.

    Args:
        file_path (str): Path to the local file

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


def get_s3_etag(bucket, key):
    """
    Get the ETag of an S3 object.

    Args:
        bucket (str): S3 bucket name
        key (str): S3 object key

    Returns:
        str: ETag of the S3 object (without quotes)
    """
    try:
        # Initialize S3 client using environment variables
        s3_client = boto3.client(
            's3',
            endpoint_url=os.environ.get("S3_ENDPOINT"),
            aws_access_key_id=os.environ.get("S3_ACCESS_KEY"),
            aws_secret_access_key=os.environ.get("S3_SECRET_KEY")
        )

        # Get object metadata
        response = s3_client.head_object(Bucket=bucket, Key=key)

        # Extract ETag (remove quotes)
        etag = response.get("ETag", "").strip('"')

        return etag
    except Exception as e:
        print(f"Error getting ETag for s3://{bucket}/{key}: {str(e)}")
        return None


def compare_local_with_s3(local_path, bucket, key):
    """
    Compare a local file with an S3 object using MD5 hash and ETag.

    Args:
        local_path (str): Path to the local file
        bucket (str): S3 bucket name
        key (str): S3 object key

    Returns:
        bool: True if the local file matches the S3 object, False otherwise
    """
    print(f"Comparing local file with S3 object:")
    print(f"Local file: {local_path}")
    print(f"S3 object: s3://{bucket}/{key}")

    # Check if local file exists
    if not os.path.exists(local_path):
        print(f"Error: Local file does not exist: {local_path}")
        return False

    # Calculate local MD5 hash
    local_md5 = calculate_local_md5(local_path)
    if local_md5 is None:
        return False

    # Get S3 ETag
    etag = get_s3_etag(bucket, key)
    if etag is None:
        return False

    # Handle multipart upload ETag (contains a dash)
    if "-" in etag:
        print(f"Detected multipart upload ETag: {etag}")
        etag_parts = etag.split("-")
        etag = etag_parts[0]  # Use only the hash part
        print(f"Using hash part of ETag: {etag}")

    # Print hashes for comparison
    print(f"\nLocal file MD5: {local_md5}")
    print(f"S3 object ETag: {etag}")

    # Compare hashes
    are_equal = etag.lower() == local_md5.lower()
    print(f"\nLocal file and S3 object are {'equal' if are_equal else 'different'} based on hash comparison.")

    # Print file size for additional information
    size = os.path.getsize(local_path)
    print(f"\nLocal file size: {size} bytes")

    return are_equal


def main():
    """Main function to handle command line arguments and execute the comparison."""
    # Check if arguments were provided
    if len(sys.argv) == 4:
        local_path = sys.argv[1]
        bucket = sys.argv[2]
        key = sys.argv[3]
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
        # Construct the S3 key (data/relative_path)
        key = f"data/{rel_path.replace('\\', '/')}"

        print(f"Using default values:")
        print(f"Local path: {local_path}")
        print(f"S3 bucket: {bucket}")
        print(f"S3 key: {key}")

    # Compare local file with S3 object
    compare_local_with_s3(local_path, bucket, key)


if __name__ == "__main__":
    main()