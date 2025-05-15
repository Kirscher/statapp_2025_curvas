"""
Dataset preparation module for the statapp application.

This module provides commands for preparing datasets for analysis.
"""

import os
import re
import time
from pathlib import Path
from typing import List, Union, Literal, Tuple

import typer
from rich.text import Text

from nnunetv2.dataset_conversion.generate_dataset_json import generate_dataset_json
from nnunetv2.experiment_planning.plan_and_preprocess_entrypoints import plan_and_preprocess
from statapp.commands.upload import upload_preprocessing_artifacts
from statapp.utils import s3_utils
from statapp.utils.progress_tracker import track_progress
from statapp.utils.utils import info, setup_logging
from statapp.core.constants import (
    DATASET_PREFIX, PATIENT_PREFIX, FILE_ENDING, CHANNEL_NAMES, LABELS,
    TRAIN_PATIENTS, VALIDATION_PATIENTS, TEST_PATIENTS
)

app = typer.Typer()

@app.command(name="download-dataset")
def download_dataset_command(
    annotator: str = typer.Argument(..., help="Annotator (1/2/3)"),
    patients: List[str] = typer.Argument(..., help="List of patient numbers (e.g., 001 034) or 'all', 'train', 'validation', 'test'"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose logging"),
) -> None:
    """
    Download a dataset for analysis without running preprocessing.

    Downloads images and annotations from S3 and organizes them for nnUNet processing.
    Creates a folder named Dataset475_CURVAS_ANNO{annotator}_{code} where code is based on the patient selection.

    Patient selection options:
    - 'all': All available patients
    - 'train': Predefined training set
    - 'validation': Predefined validation set
    - 'test': Predefined test set
    - Custom list: Specific patient numbers (e.g., 001 034)

    Use --verbose to enable verbose logging output.
    """
    # Set up logger
    logger = setup_logging(verbose)

    # Validate annotator input
    if annotator not in ["1", "2", "3"]:
        logger.error(f"Annotator must be 1, 2, or 3. Got {annotator}")
        return

    # Handle patient selection
    patient_selection = patients
    if len(patients) == 1:
        # Check if it's one of the predefined sets
        if patients[0] in ["all", "train", "validation", "test"]:
            patient_selection = patients[0]

    # Download dataset
    dataset_dir, selected_patients = download_dataset(
        annotator=annotator,
        patients=patient_selection,
        verbose=verbose,
        skip=False
    )

    if not dataset_dir or not selected_patients:
        logger.error("Failed to download dataset")
        return

    logger.info(Text(f"Dataset downloaded to {dataset_dir}", style="bold green"))

@app.command(name="download-preprocessing")
def download_preprocessing_command(
    annotator: str = typer.Argument(..., help="Annotator (1/2/3)"),
    patients: List[str] = typer.Argument(..., help="List of patient numbers (e.g., 001 034) or 'all', 'train', 'validation', 'test'"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose logging"),
) -> None:
    """
    Download preprocessing artifacts for a dataset.

    Checks if preprocessing artifacts exist in S3 for the specified dataset and downloads them if they do.

    Patient selection options:
    - 'all': All available patients
    - 'train': Predefined training set
    - 'validation': Predefined validation set
    - 'test': Predefined test set
    - Custom list: Specific patient numbers (e.g., 001 034)

    Use --verbose to enable verbose logging output.
    """
    # Set up logger
    logger = setup_logging(verbose)

    # Validate annotator input
    if annotator not in ["1", "2", "3"]:
        logger.error(f"Annotator must be 1, 2, or 3. Got {annotator}")
        return

    # Handle patient selection
    patient_selection = patients
    if len(patients) == 1:
        # Check if it's one of the predefined sets
        if patients[0] in ["all", "train", "validation", "test"]:
            patient_selection = patients[0]

    # Generate dataset code for folder naming
    dataset_code = get_dataset_code(patient_selection)

    # Check if preprocessing artifacts exist in S3 and download them
    preprocessing_exists = download_preprocessing(
        annotator=annotator,
        dataset_code=dataset_code,
        verbose=verbose
    )

    if preprocessing_exists:
        logger.info(Text("Preprocessing artifacts downloaded successfully", style="bold green"))
    else:
        logger.error(Text("No preprocessing artifacts found for this dataset", style="bold red"))

def get_dataset_code(patients: Union[List[str], Literal["all", "train", "validation", "test"]]) -> str:
    """
    Generate a dataset code based on the patient selection.

    Args:
        patients: List of patient numbers or a predefined set ("all", "train", "validation", "test")

    Returns:
        str: Dataset code for folder naming
    """
    if isinstance(patients, str):
        if patients in ["all", "train", "validation", "test"]:
            return patients

    # For custom patient lists, join the numbers
    return "".join(sorted(patients))

def download_dataset(
    annotator: str,
    patients: Union[List[str], Literal["all", "train", "validation", "test"]],
    verbose: bool = False,
    skip: bool = False
) -> Tuple[Path, List[str]]:
    """
    Download a dataset for analysis.

    Args:
        annotator: Annotator (1/2/3)
        patients: List of patient numbers or a predefined set ("all", "train", "validation", "test")
        verbose: Enable verbose logging
        skip: Skip download and only set up directories

    Returns:
        Tuple[Path, List[str]]: Dataset directory path and list of selected patients
    """
    # Set up logger
    logger = setup_logging(verbose)

    # Resolve patient selection
    selected_patients = []

    # List contents of the data directory
    contents = s3_utils.list_data_directory()

    # Extract folder names that match the pattern UKCHLL[NNN]
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

    # Determine which patients to process based on the selection
    if patients == "all":
        selected_patients = list(available_patients.keys())
        info(Text(f"Selected all {len(selected_patients)} patients", style="bold green"))
    elif patients == "train":
        for patient in TRAIN_PATIENTS:
            if patient in available_patients:
                selected_patients.append(patient)
        info(Text(f"Selected {len(selected_patients)} training patients", style="bold green"))
    elif patients == "validation":
        for patient in VALIDATION_PATIENTS:
            if patient in available_patients:
                selected_patients.append(patient)
        info(Text(f"Selected {len(selected_patients)} validation patients", style="bold green"))
    elif patients == "test":
        for patient in TEST_PATIENTS:
            if patient in available_patients:
                selected_patients.append(patient)
        info(Text(f"Selected {len(selected_patients)} test patients", style="bold green"))
    else:
        # Custom list of patients
        missing_patients = []
        for patient in patients:
            if patient in available_patients:
                selected_patients.append(patient)
            else:
                missing_patients.append(patient)

        if missing_patients:
            missing_list = ", ".join([f"UKCHLL{p}" for p in missing_patients])
            info(Text.assemble(("Warning: ", "bold yellow"), (f"The following patients were not found: {missing_list}", "")))

    if not selected_patients:
        info(Text.assemble(("Error: ", "bold red"), ("No valid patients selected", "")))
        return None, []

    logger.info(Text(f"Processing {len(selected_patients)} patients with annotator {annotator}", style="bold green"))

    # Generate dataset code
    dataset_code = get_dataset_code(patients)

    # Create necessary directories
    nnunet_raw_dir = Path("nnUNet_raw")
    dataset_dir = nnunet_raw_dir / f"{DATASET_PREFIX}{annotator}_{dataset_code}"
    images_dir = dataset_dir / "imagesTr"
    labels_dir = dataset_dir / "labelsTr"

    os.makedirs(nnunet_raw_dir, exist_ok=True)
    os.makedirs(dataset_dir, exist_ok=True)
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)

    if skip:
        info(Text("Skipping download as requested with --skip option", style="bold yellow"))
        return dataset_dir, selected_patients

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
            missing_files = []
            if not image_exists:
                missing_files.append(f"image for UKCHLL{patient}")
            if not annotation_exists:
                missing_files.append(f"annotation {annotator} for UKCHLL{patient}")
            if missing_files:
                info(Text.assemble(("Warning: ", "bold yellow"), (f"Missing files: {', '.join(missing_files)}", "")))

    if not files_to_download:
        info(Text.assemble(("Error: ", "bold red"), ("No valid files to download", "")))
        return None, []

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
            info(Text(f"Error processing {display_name}: {str(e)}", style="bold red"))

    # Track progress and process files
    track_progress(files_to_download, get_file_size, process_file)

    # Count the number of unique patients (each patient has both image and annotation)
    num_patients = len(set(file_info['patient'] for file_info in files_to_download))

    # Generate dataset.json file
    generate_dataset_json(
        output_folder=str(dataset_dir),
        channel_names=CHANNEL_NAMES,
        labels=LABELS,
        num_training_cases=num_patients,
        file_ending=FILE_ENDING
    )

    # Group all directory information into a single message
    info(Text.assemble(
        ("Dataset preparation complete!", "bold green"),
        ("\nDataset directory: ", "bold"), (f"{dataset_dir}", ""),
        ("\nImages directory: ", "bold"), (f"{images_dir}", ""),
        ("\nLabels directory: ", "bold"), (f"{labels_dir}", ""),
        ("\nDataset JSON file: ", "bold"), (f"{dataset_dir / 'dataset.json'}", "")
    ))

    return dataset_dir, selected_patients

def download_preprocessing(
    annotator: str,
    dataset_code: str,
    verbose: bool = False
) -> bool:
    """
    Check if preprocessing artifacts exist in S3 and download them if they do.

    Args:
        annotator: Annotator (1/2/3)
        dataset_code: Dataset code for folder naming
        verbose: Enable verbose logging

    Returns:
        bool: True if preprocessing artifacts were downloaded, False otherwise
    """
    # Set up logger
    logger = setup_logging(verbose)

    # Construct the preprocessing folder name
    preprocessing_folder = f"{DATASET_PREFIX}{annotator}_{dataset_code}"

    # Check if preprocessing artifacts exist in S3
    artifacts_contents = s3_utils.list_artifacts_directory()
    preprocessing_path = f"{os.environ.get('S3_PROPROCESSING_ARTIFACTS_SUBDIR', 'preprocessing')}/{preprocessing_folder}"

    # Check if any files exist with the preprocessing path prefix
    preprocessing_exists = any(
        item['Key'].startswith(f"{os.environ['S3_ARTIFACTS_DIR']}/{preprocessing_path}/") 
        for item in artifacts_contents
    )

    if not preprocessing_exists:
        logger.info(f"No preprocessing artifacts found for {preprocessing_folder}")
        return False

    # Download preprocessing artifacts
    logger.info(f"Downloading preprocessing artifacts for {preprocessing_folder}")

    # Construct the remote path
    remote_path = f"{os.environ['S3_BUCKET']}/{os.environ['S3_ARTIFACTS_DIR']}/{preprocessing_path}"

    # Construct the local path
    local_path = f"nnUNet_preprocessed/{DATASET_PREFIX}{annotator}_{dataset_code}"

    # Ensure the local directory exists
    os.makedirs(local_path, exist_ok=True)

    # Collect all objects to download
    files_to_download = []
    bucket = os.environ['S3_BUCKET']
    prefix = f"{os.environ['S3_ARTIFACTS_DIR']}/{preprocessing_path}"

    # Find all files in the preprocessing directory
    for item in artifacts_contents:
        key = item['Key']
        if key.startswith(prefix) and not key.endswith('/'):
            # Calculate the relative path from the prefix
            rel_path = key[len(prefix):].lstrip('/')

            # Construct the local file path
            local_file_path = os.path.join(local_path, rel_path)

            # Add to the list of files to download
            files_to_download.append({
                'remote_path': f"{bucket}/{key}",
                'local_path': local_file_path,
                'display_name': os.path.basename(key)
            })

    if not files_to_download:
        logger.warning("No preprocessing artifact files found to download")
        return False

    # Function to get file size
    def get_file_size(file_info):
        # Get the actual file size from S3 for a single file
        bucket, key = s3_utils.parse_remote_path(file_info['remote_path'])
        return s3_utils.get_file_size(bucket, key)

    # Function to process each file
    def process_file(file_info, progress_tracker):
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

            # Ensure the directory exists
            os.makedirs(os.path.dirname(file_info['local_path']), exist_ok=True)

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
            logger.error(f"Error processing {display_name}: {str(e)}")

    # Track progress and process files
    track_progress(files_to_download, get_file_size, process_file)

    # Count the number of successfully downloaded files
    downloaded_files = [f for f in files_to_download if os.path.exists(f['local_path'])]

    if downloaded_files:
        logger.info(f"Downloaded {len(downloaded_files)} preprocessing artifact files")
        return True
    else:
        logger.warning("Failed to download preprocessing artifacts")
        return False

@app.command()
def prepare(
    annotator: str = typer.Argument(..., help="Annotator (1/2/3)"),
    patients: List[str] = typer.Argument(..., help="List of patient numbers (e.g., 001 034) or 'all', 'train', 'validation', 'test'"),
    skip: bool = typer.Option(False, "--skip", help="Skip download and only run preprocessing"),
    num_processes_fingerprint: int = typer.Option(2, "--num-processes-fingerprint", "-npfp", help="Number of processes to use for fingerprint extraction"),
    num_processes: int = typer.Option(2, "--num-processes", "-np", help="Number of processes to use for preprocessing"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose logging"),
) -> None:
    """
    Prepare a dataset for analysis in the S3 data folder.

    Downloads images and annotations from S3 and organizes them for nnUNet processing.
    Then runs nnUNet planning and preprocessing for dataset 475 with 3d_fullres configuration.

    Patient selection options:
    - 'all': All available patients
    - 'train': Predefined training set
    - 'validation': Predefined validation set
    - 'test': Predefined test set
    - Custom list: Specific patient numbers (e.g., 001 034)

    Use --skip to skip the download part and only run the path setup and preprocessing.
    Use --num-processes-fingerprint to set the number of processes for fingerprint extraction (default: 2).
    Use --num-processes to set the number of processes for preprocessing (default: 2).
    Use --verbose to enable verbose logging output.
    """
    # Set up logger
    logger = setup_logging(verbose)

    # Validate annotator input
    if annotator not in ["1", "2", "3"]:
        logger.error(f"Annotator must be 1, 2, or 3. Got {annotator}")
        return

    # Handle patient selection
    patient_selection = patients
    if len(patients) == 1:
        # Check if it's one of the predefined sets
        if patients[0] in ["all", "train", "validation", "test"]:
            patient_selection = patients[0]

    # Generate dataset code for folder naming
    dataset_code = get_dataset_code(patient_selection)

    # Download dataset (or just set up directories if skip=True)
    dataset_dir, selected_patients = download_dataset(
        annotator=annotator,
        patients=patient_selection,
        verbose=verbose,
        skip=skip
    )

    if not dataset_dir or not selected_patients:
        logger.error("Failed to prepare dataset")
        return

    # Check if preprocessing artifacts already exist in S3
    preprocessing_exists = download_preprocessing(
        annotator=annotator,
        dataset_code=dataset_code,
        verbose=verbose
    )

    if preprocessing_exists:
        logger.info("Using downloaded preprocessing artifacts")
    else:
        # Run plan_and_preprocess
        logger.info("Running nnUNet planning and preprocessing...")
        plan_and_preprocess(
            dataset_ids=[475],
            configurations=["3d_fullres"],
            verify_dataset_integrity=True,
            num_processes_fingerprint=num_processes_fingerprint,
            num_processes=[num_processes],
            logger=logger
        )

        logger.info("nnUNet planning and preprocessing complete!")

        # Upload preprocessing artifacts to S3
        logger.info("Uploading preprocessing artifacts to S3...")

        # Construct the preprocessing folder name
        preprocessing_folder = f"{DATASET_PREFIX}{annotator}_{dataset_code}"

        # Upload the preprocessing artifacts
        upload_preprocessing_artifacts(
            directory=f"nnUNet_preprocessed/{DATASET_PREFIX}{annotator}_{dataset_code}",
            preprocessingfolder=preprocessing_folder,
            verbose=verbose
        )

        logger.info("Preprocessing artifacts uploaded to S3")

    logger.info(Text("Dataset preparation complete!", style="bold green"))
