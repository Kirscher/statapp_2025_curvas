"""
Dataset preparation module for the statapp application.

This module provides commands for preparing datasets for analysis.
"""

import os
import re
import time
from pathlib import Path
from rich.text import Text
import typer
from typing import Optional, List, Dict, Any

from statapp.utils import s3_utils
from statapp.utils.progress_tracker import ProgressTracker, track_progress
from statapp.utils.utils import setup_nnunet_env, info, pretty_print, create_progress
from nnunetv2.dataset_conversion.generate_dataset_json import generate_dataset_json

# Constants
NNUNET_RAW_DIR = "nnUNet_raw"
DATASET_PREFIX = "Dataset475_CURVAS_ANNO"
PATIENT_PREFIX = "CURVAS"
FILE_ENDING = ".nii.gz"
CHANNEL_NAMES = {
    "0": "CT"
}
LABELS = {
    "background": 0,
    "pancreas": 1,
    "kidney": 2,
    "liver": 3
}

app = typer.Typer()

@app.command()
def prepare(
    annotator: str = typer.Argument(..., help="Annotator (1/2/3)"),
    patients: List[str] = typer.Argument(..., help="List of patient numbers (e.g., 001 034) or all for all"),
) -> None:
    """
    Prepare a dataset for analysis in the S3 data folder.

    Downloads images and annotations from S3 and organizes them for nnUNet processing.
    """
    # Validate annotator input
    if annotator not in ["1", "2", "3"]:
        pretty_print(f"[bold red]Error:[/bold red] Annotator must be 1, 2, or 3. Got {annotator}")
        return

    # List contents of the data directory
    contents = s3_utils.list_data_directory()

    # Extract folder names that match the pattern UKCHLL[NNN]
    # The pattern should match only keys that are exactly in the format data/UKCHLL001 or data/UKCHLL001/
    # It should not match keys like data/UKCHLLLICENSE or data/UKCHLLpyproject.toml
    pattern = re.compile(r'^' + re.escape(os.environ['S3_DATA_DIR']) + r'/UKCHLL(\d{3})(?:/.*)?$')
    available_patients = {}

    for item in contents:
        key = item['Key']
        match = pattern.match(key)
        if match:
            patient_num = match.group(1)
            if patient_num not in available_patients:
                available_patients[patient_num] = []
            available_patients[patient_num].append(key)

    # Determine which patients to process
    selected_patients = []
    if len(patients) == 1 and patients[0] == "all":
        selected_patients = list(available_patients.keys())
        pretty_print(f"[bold green]Selected all {len(selected_patients)} patients[/bold green]")
    else:
        for patient in patients:
            if patient in available_patients:
                selected_patients.append(patient)
            else:
                pretty_print(f"[bold yellow]Warning:[/bold yellow] Patient UKCHLL{patient} not found in S3 data")

    if not selected_patients:
        pretty_print("[bold red]Error:[/bold red] No valid patients selected")
        return

    pretty_print(f"[bold green]Processing {len(selected_patients)} patients with annotator {annotator}[/bold green]")

    # Create necessary directories
    nnunet_raw_dir = Path(NNUNET_RAW_DIR)
    dataset_dir = nnunet_raw_dir / f"{DATASET_PREFIX}{annotator}"
    images_dir = dataset_dir / "imagesTr"
    labels_dir = dataset_dir / "labelsTr"

    os.makedirs(nnunet_raw_dir, exist_ok=True)
    os.makedirs(dataset_dir, exist_ok=True)
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)

    # Setup environment variables
    setup_nnunet_env(Path.cwd(), Path(NNUNET_RAW_DIR), Path("nnUNet_preprocessed"), Path("nnUNet_results"))

    # Prepare file list for downloading
    files_to_download = []
    bucket = os.environ['S3_BUCKET']

    for patient in selected_patients:
        # Find the image and annotation files for this patient
        patient_prefix = f"{os.environ['S3_DATA_DIR']}/UKCHLL{patient}"
        image_key = f"{patient_prefix}/image.nii.gz"
        annotation_key = f"{patient_prefix}/annotation_{annotator}.nii.gz"

        # Check if the files exist
        image_exists = any(item['Key'] == image_key for item in contents)
        annotation_exists = any(item['Key'] == annotation_key for item in contents)

        if image_exists and annotation_exists:
            # Create separate entries for image and annotation files
            files_to_download.append({
                'type': 'image',
                'patient': patient,
                'remote_path': f"{bucket}/{image_key}",
                'local_path': str(images_dir / f"{PATIENT_PREFIX}_{patient}_0000{FILE_ENDING}"),
                'display_name': f"image for UKCHLL{patient}"
            })

            files_to_download.append({
                'type': 'annotation',
                'patient': patient,
                'remote_path': f"{bucket}/{annotation_key}",
                'local_path': str(labels_dir / f"{PATIENT_PREFIX}_{patient}{FILE_ENDING}"),
                'display_name': f"annotation {annotator} for UKCHLL{patient}"
            })
        else:
            if not image_exists:
                pretty_print(f"[bold yellow]Warning:[/bold yellow] Image file for UKCHLL{patient} not found")
            if not annotation_exists:
                pretty_print(f"[bold yellow]Warning:[/bold yellow] Annotation {annotator} for UKCHLL{patient} not found")

    if not files_to_download:
        pretty_print("[bold red]Error:[/bold red] No valid files to download")
        return

    # Function to get file size (now uses actual S3 file size)
    def get_file_size(file_info):
        # Get the actual file size from S3 for a single file
        bucket, key = s3_utils.parse_remote_path(file_info['remote_path'])
        return s3_utils.get_file_size(bucket, key)

    # Function to process each file
    def process_file(file_info, progress_tracker):
        patient = file_info['patient']
        file_type = file_info['type']
        display_name = file_info['display_name']

        # Get the file size
        file_size = get_file_size(file_info)

        # Record start time
        start_time = time.time()

        try:
            # Start tracking file download
            progress_tracker.start_file(
                file_info, 
                f"Downloading {display_name}", 
                file_size
            )

            # Download file
            success = s3_utils.download_file(
                file_info['remote_path'],
                file_info['local_path'],
                callback=progress_tracker.get_progress_callback(
                    f"Downloading {display_name}",
                    file_size,
                    start_time
                )
            )

            # Complete file download
            progress_tracker.complete_file(
                f"Downloading {display_name}",
                file_size,
                start_time,
                success=success
            )

            if not success:
                raise Exception(f"Failed to download {display_name}")

        except Exception as e:
            # Mark file as failed
            progress_tracker.complete_file(
                f"Error downloading {display_name}",
                file_size,
                start_time,
                success=False
            )
            pretty_print(f"[bold red]Error processing {display_name}: {str(e)}[/bold red]")

    # Track progress and process files
    track_progress(files_to_download, get_file_size, process_file)

    # Generate dataset.json file
    pretty_print(f"[bold]Generating dataset.json file...[/bold]")

    # Count the number of unique patients (each patient has both image and annotation)
    num_patients = len(set(file_info['patient'] for file_info in files_to_download))

    generate_dataset_json(
        output_folder=str(dataset_dir),
        channel_names=CHANNEL_NAMES,
        labels=LABELS,
        num_training_cases=num_patients,
        file_ending=FILE_ENDING
    )

    pretty_print(f"[bold green]Dataset preparation complete![/bold green]")
    pretty_print(f"[bold]Dataset directory: {dataset_dir}[/bold]")
    pretty_print(f"[bold]Images directory: {images_dir}[/bold]")
    pretty_print(f"[bold]Labels directory: {labels_dir}[/bold]")
    pretty_print(f"[bold]Dataset JSON file: {dataset_dir / 'dataset.json'}[/bold]")