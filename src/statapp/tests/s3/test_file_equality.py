"""
Test file equality by comparing hash codes.

This script calculates and compares MD5 hashes of two files to determine if they are equal.
It can be used to debug issues with file comparison in the S3Singleton implementation.

Usage:
    python -m statapp.tests.s3.test_file_equality <file1_path> <file2_path>
    
    If no arguments are provided, it will prompt for file paths interactively.
"""

import hashlib
import os
import sys
from pathlib import Path


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
    except FileNotFoundError:
        print(f"Error: File not found: {file_path}")
        return None
    except PermissionError:
        print(f"Error: Permission denied for file: {file_path}")
        return None
    except Exception as e:
        print(f"Error calculating hash for {file_path}: {str(e)}")
        return None


def compare_files(file1_path, file2_path):
    """
    Compare two files by their MD5 hashes.
    
    Args:
        file1_path (str): Path to the first file
        file2_path (str): Path to the second file
        
    Returns:
        bool: True if files have the same hash, False otherwise
    """
    print(f"Comparing files:")
    print(f"File 1: {file1_path}")
    print(f"File 2: {file2_path}")
    
    # Check if files exist
    if not os.path.exists(file1_path):
        print(f"Error: File 1 does not exist: {file1_path}")
        return False
    if not os.path.exists(file2_path):
        print(f"Error: File 2 does not exist: {file2_path}")
        return False
    
    # Calculate hashes
    hash1 = calculate_md5(file1_path)
    hash2 = calculate_md5(file2_path)
    
    if hash1 is None or hash2 is None:
        return False
    
    # Print hashes
    print(f"\nFile 1 MD5: {hash1}")
    print(f"File 2 MD5: {hash2}")
    
    # Compare hashes
    are_equal = hash1.lower() == hash2.lower()
    print(f"\nFiles are {'equal' if are_equal else 'different'} based on MD5 hash.")
    
    # Print file sizes for additional information
    size1 = os.path.getsize(file1_path)
    size2 = os.path.getsize(file2_path)
    print(f"\nFile 1 size: {size1} bytes")
    print(f"File 2 size: {size2} bytes")
    
    return are_equal


def main():
    """Main function to handle command line arguments and execute the comparison."""
    # Check if arguments were provided
    if len(sys.argv) == 3:
        file1_path = sys.argv[1]
        file2_path = sys.argv[2]
    else:
        # If no arguments, prompt for file paths
        print("Enter paths for files to compare:")
        file1_path = input("File 1 path: ").strip()
        file2_path = input("File 2 path: ").strip()
    
    # Compare files
    compare_files(file1_path, file2_path)


if __name__ == "__main__":
    main()