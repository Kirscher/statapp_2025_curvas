"""
Test the s3_utils module.

This script tests the s3_utils module by downloading a file from S3 and verifying
that it was downloaded correctly.

Usage:
    python -m statapp.tests.s3.test_s3_utils
"""

import hashlib
import os
import tempfile
import time

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Import the s3_utils module
from statapp.utils import s3_utils
from statapp.utils.utils import pretty_print


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
        pretty_print(f"[bold red]Error calculating hash for {file_path}: {str(e)}[/bold red]")
        return None


def test_download_file():
    """
    Test downloading a file from S3 using s3_utils.download_file.
    """
    pretty_print("[bold]Testing s3_utils.download_file[/bold]")
    
    # Create a temporary directory for the test
    with tempfile.TemporaryDirectory() as temp_dir:
        # Get the first file from the data directory
        contents = s3_utils.list_data_directory()
        if not contents:
            pretty_print("[bold red]Error: No files found in data directory[/bold red]")
            return False
        
        # Find a .nii.gz file
        test_key = None
        for item in contents:
            if item['Key'].endswith('.nii.gz'):
                test_key = item['Key']
                break
        
        if not test_key:
            pretty_print("[bold red]Error: No .nii.gz files found in data directory[/bold red]")
            return False
        
        # Prepare paths
        bucket = os.environ.get("S3_BUCKET", "projet-statapp-segmedic")
        remote_path = f"{bucket}/{test_key}"
        local_path = os.path.join(temp_dir, os.path.basename(test_key))
        
        pretty_print(f"[bold]Downloading file:[/bold] {remote_path} -> {local_path}")
        
        # Define a simple progress callback
        def progress_callback(bytes_transferred):
            # This is just a simple callback for testing
            pass
        
        # Download the file
        start_time = time.time()
        success = s3_utils.download_file(remote_path, local_path, callback=progress_callback)
        end_time = time.time()
        
        if not success:
            pretty_print("[bold red]Error: Failed to download file[/bold red]")
            return False
        
        # Verify the file exists
        if not os.path.exists(local_path):
            pretty_print("[bold red]Error: Downloaded file does not exist[/bold red]")
            return False
        
        # Get file size
        file_size = os.path.getsize(local_path)
        
        pretty_print(f"[bold green]Success![/bold green] Downloaded {file_size} bytes in {end_time - start_time:.2f} seconds")
        return True


def test_download_directory():
    """
    Test downloading a directory from S3 using s3_utils.download_directory.
    """
    pretty_print("[bold]Testing s3_utils.download_directory[/bold]")
    
    # Create a temporary directory for the test
    with tempfile.TemporaryDirectory() as temp_dir:
        # Get a patient directory
        contents = s3_utils.list_data_directory()
        if not contents:
            pretty_print("[bold red]Error: No files found in data directory[/bold red]")
            return False
        
        # Find a patient directory
        patient_prefix = None
        for item in contents:
            key = item['Key']
            parts = key.split('/')
            if len(parts) >= 2 and parts[1].startswith('UKCHLL'):
                patient_prefix = f"{parts[0]}/{parts[1]}"
                break
        
        if not patient_prefix:
            pretty_print("[bold red]Error: No patient directories found[/bold red]")
            return False
        
        # Prepare paths
        bucket = os.environ.get("S3_BUCKET", "projet-statapp-segmedic")
        remote_path = f"{bucket}/{patient_prefix}"
        
        pretty_print(f"[bold]Downloading directory:[/bold] {remote_path} -> {temp_dir}")
        
        # Define a simple progress callback
        def progress_callback(bytes_transferred, filename):
            # This is just a simple callback for testing
            pass
        
        # Download the directory
        start_time = time.time()
        downloaded_files = s3_utils.download_directory(remote_path, temp_dir, callback=progress_callback)
        end_time = time.time()
        
        if not downloaded_files:
            pretty_print("[bold red]Error: No files downloaded[/bold red]")
            return False
        
        # Verify the files exist
        for file_path in downloaded_files:
            if not os.path.exists(file_path):
                pretty_print(f"[bold red]Error: Downloaded file does not exist: {file_path}[/bold red]")
                return False
        
        pretty_print(f"[bold green]Success![/bold green] Downloaded {len(downloaded_files)} files in {end_time - start_time:.2f} seconds")
        for file_path in downloaded_files:
            pretty_print(f"  - {os.path.basename(file_path)} ({os.path.getsize(file_path)} bytes)")
        
        return True


def main():
    """Main function to run the tests."""
    pretty_print("[bold]Testing s3_utils module[/bold]")
    
    # Test download_file
    if test_download_file():
        pretty_print("[bold green]download_file test passed[/bold green]")
    else:
        pretty_print("[bold red]download_file test failed[/bold red]")
    
    # Test download_directory
    if test_download_directory():
        pretty_print("[bold green]download_directory test passed[/bold green]")
    else:
        pretty_print("[bold red]download_directory test failed[/bold red]")


if __name__ == "__main__":
    main()