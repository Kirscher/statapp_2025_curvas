"""
Ensemble module for the statapp application.

This module provides a command to ensemble predictions from multiple models.
"""

import os
import re
import tempfile
import threading
import time
from pathlib import Path
from typing import List, Union, Literal, Optional

import typer
from rich.text import Text

from statapp.utils import s3_utils
from statapp.utils.progress_tracker import track_progress
from statapp.utils.utils import info, setup_logging
from statapp.utils.upload_utils import upload_directory_to_s3
from statapp.core.constants import TRAIN_PATIENTS, VALIDATION_PATIENTS, TEST_PATIENTS
from nnunetv2.ensembling.ensemble import ensemble_folders

app = typer.Typer()

def get_selected_patients(patients: Union[List[str], Literal["all", "train", "validation", "test"]], verbose: bool = False) -> List[str]:
    """
    Get a list of selected patients based on the input.

    Args:
        patients: List of patient numbers or a predefined set ("all", "train", "validation", "test")
        verbose (bool): Enable verbose logging

    Returns:
        List[str]: List of selected patient numbers
    """
    # Set up logger
    logger = setup_logging(verbose)

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
    selected_patients = []

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
        return []

    return selected_patients

def get_available_models() -> List[str]:
    """
    Get a list of available models from S3.

    Returns:
        List[str]: List of model folder names
    """
    # List contents of the models directory
    contents = s3_utils.list_artifacts_directory()

    # Extract folder names that match the pattern anno[ANNOTATOR]_init[SEED]_foldall
    pattern = re.compile(r'^' + re.escape(os.environ['S3_ARTIFACTS_DIR']) + r'/' + 
                         re.escape(os.environ['S3_MODEL_ARTIFACTS_SUBDIR']) + 
                         r'/anno(\d+)_init(\d+)_foldall(?:/.*)?$')

    available_models = set()

    for item in contents:
        key = item['Key']
        match = pattern.match(key)
        if match:
            # Extract the model folder name (anno[ANNOTATOR]_init[SEED]_foldall)
            model_path = key.split('/')
            if len(model_path) >= 3:
                model_name = model_path[2]  # Get the third component (index 2)
                available_models.add(model_name)

    return sorted(list(available_models))

def download_curvas_files(patient_id: str, model_name: str, output_dir: Path, verbose: bool = False, progress_tracker = None) -> bool:
    """
    Download all files for a patient and model.

    Args:
        patient_id (str): Patient ID (e.g., 001)
        model_name (str): Model name (e.g., anno1_init112233_foldall)
        output_dir (Path): Output directory to save the files
        verbose (bool): Enable verbose logging
        progress_tracker: Optional progress tracker instance

    Returns:
        bool: True if files were downloaded successfully, False otherwise
    """
    logger = setup_logging(verbose)

    # Create patient and model directories
    patient_dir = output_dir / f"UKCHLL{patient_id}"
    model_dir = patient_dir / model_name
    os.makedirs(model_dir, exist_ok=True)

    # Define the remote path
    remote_dir = f"{os.environ['S3_OUTPUT_DIR']}/UKCHLL{patient_id}/{model_name}"

    # List contents of the output directory
    contents = s3_utils.list_output_directory()

    # Find all files for this patient and model
    patient_files = []
    pattern = re.compile(r'^' + re.escape(remote_dir) + r'/')

    for item in contents:
        key = item['Key']
        if pattern.match(key):
            patient_files.append(key)

    if not patient_files:
        logger.error(f"No files found for patient UKCHLL{patient_id} and model {model_name}")
        return False

    # Download the files
    success = True

    # Start time for tracking
    start_time = time.time()

    # Download each file
    for file_path in patient_files:
        # Get the file name from the path
        file_name = os.path.basename(file_path)

        # Get the file size
        bucket, key = s3_utils.parse_remote_path(file_path)
        file_size = s3_utils.get_file_size(bucket, key)

        # Create the local path
        local_file_path = model_dir / file_name

        logger.info(f"Downloading {file_name} for patient UKCHLL{patient_id} and model {model_name}...")

        # Create a callback function for progress updates
        def file_progress_callback(bytes_transferred):
            if progress_tracker:
                progress_tracker.update_file_progress(
                    bytes_transferred, 
                    file_size, 
                    f"Downloading {file_name} for patient UKCHLL{patient_id} and model {model_name}", 
                    start_time
                )

        # Download the file with progress tracking
        file_success = s3_utils.download_file(
            remote_path=file_path,
            local_path=str(local_file_path),
            callback=file_progress_callback
        )

        if not file_success:
            logger.error(f"Failed to download {file_name} for patient UKCHLL{patient_id} and model {model_name}")
            success = False

    logger.info(f"All files downloaded successfully for patient UKCHLL{patient_id} and model {model_name}")
    return success

@app.command()
def dl_ensemble(
    patients: List[str] = typer.Argument(..., help="List of patient numbers (e.g., 001 034) or 'all', 'train', 'validation', 'test'"),
    models: List[str] = typer.Option(["all"], "-m", "--models", help="List of models to use for ensembling (e.g., anno1_init112233_foldall) or 'all'"),
    nb_workers: int = typer.Option(10, '-j', "--jobs", help="Number of processes to run"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose logging")
) -> None:
    """
    Download predictions from multiple models.

    Downloads CURVAS_*.pkl and CURVAS_*.npz files for each patient and model,
    and saves them to an 'ensembling' folder.

    Patient selection options:
    - 'all': All available patients
    - 'train': Predefined training set
    - 'validation': Predefined validation set
    - 'test': Predefined test set
    - Custom list: Specific patient numbers (e.g., 001 034)

    Model selection options:
    - 'all': All available models
    - Custom list: Specific model names (e.g., anno1_init112233_foldall)
    """
    # Set up logger
    logger = setup_logging(verbose)

    # Handle patient selection
    patient_selection = patients
    if len(patients) == 1:
        # Check if it's one of the predefined sets
        if patients[0] in ["all", "train", "validation", "test"]:
            patient_selection = patients[0]

    # Get selected patients
    selected_patients = get_selected_patients(patient_selection, verbose)
    if not selected_patients:
        return

    # Get available models
    available_models = get_available_models()
    if not available_models:
        logger.error("No models found in S3")
        return

    # Handle model selection
    selected_models = []
    if len(models) == 1 and models[0] == "all":
        selected_models = available_models
        info(Text(f"Selected all {len(selected_models)} models", style="bold green"))
    else:
        # Custom list of models
        missing_models = []
        for model in models:
            if model in available_models:
                selected_models.append(model)
            else:
                missing_models.append(model)

        if missing_models:
            missing_list = ", ".join(missing_models)
            info(Text.assemble(("Warning: ", "bold yellow"), (f"The following models were not found: {missing_list}", "")))

    if not selected_models:
        info(Text.assemble(("Error: ", "bold red"), ("No valid models selected", "")))
        return

    # Create ensembling directory
    ensembling_dir = Path("ensembling")
    os.makedirs(ensembling_dir, exist_ok=True)

    # Process each patient and model
    logger.info(f"Processing {len(selected_patients)} patients with {len(selected_models)} models")

    # Define a function to count the number of files for a patient-model pair
    def count_pair_files(pair):
        patient_id, model_name = pair

        # Define the remote path
        remote_dir = f"{os.environ['S3_OUTPUT_DIR']}/UKCHLL{patient_id}/{model_name}"

        # List contents of the output directory
        contents = s3_utils.list_output_directory()

        # Find all files for this patient and model
        pattern = re.compile(r'^' + re.escape(remote_dir) + r'/')
        file_count = 0

        for item in contents:
            key = item['Key']
            if pattern.match(key):
                file_count += 1

        return file_count

    # Create a list of patient-model pairs to process
    pairs = [(patient, model) for patient in selected_patients for model in selected_models]

    # Calculate the total number of files to download
    total_files = sum(count_pair_files(pair) for pair in pairs)

    logger.info(f"This will download a total of {total_files} files across {len(pairs)} patient-model pairs")

    # Define a function to get the combined size of all files for a patient-model pair
    def get_pair_size(pair):
        patient_id, model_name = pair

        # Define the remote path
        remote_dir = f"{os.environ['S3_OUTPUT_DIR']}/UKCHLL{patient_id}/{model_name}"

        # List contents of the output directory
        contents = s3_utils.list_output_directory()

        # Find all files for this patient and model
        pattern = re.compile(r'^' + re.escape(remote_dir) + r'/')
        total_size = 0

        for item in contents:
            key = item['Key']
            if pattern.match(key):
                # Get the file size
                bucket, file_key = s3_utils.parse_remote_path(key)
                file_size = s3_utils.get_file_size(bucket, file_key)
                total_size += file_size

        return total_size

    # Define a function to process a patient-model pair
    def process_pair(pair, progress_tracker):
        patient_id, model_name = pair
        # Get the actual size of the pair
        pair_size = get_pair_size(pair)
        progress_tracker.start_file(pair, f"Processing patient UKCHLL{patient_id} with model {model_name}", pair_size)
        logger.info(f"Processing patient UKCHLL{patient_id} with model {model_name}...")

        try:
            # Download CURVAS files
            success = download_curvas_files(patient_id, model_name, ensembling_dir, verbose, progress_tracker)

            if success:
                progress_tracker.complete_file(f"Processed patient UKCHLL{patient_id} with model {model_name}", pair_size, time.time(), success=True)
            else:
                progress_tracker.complete_file(f"Failed to process patient UKCHLL{patient_id} with model {model_name}", pair_size, time.time(), success=False)

        except Exception as e:
            logger.error(f"Error processing patient UKCHLL{patient_id} with model {model_name}: {str(e)}")
            progress_tracker.complete_file(f"Error processing patient UKCHLL{patient_id} with model {model_name}", pair_size, time.time(), success=False)

    # Track progress of processing pairs
    track_progress(pairs, get_pair_size, process_pair, total_files)

    logger.info(Text("Ensembling completed successfully", style="bold green"))


@app.command()
def run_ensemble(
    patients: List[str] = typer.Argument(..., help="List of patient numbers (e.g., 001 034)"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose logging"),
    num_processes: int = typer.Option(10, '-j', "--jobs", help="Number of processes to run")
) -> None:
    """
    Run ensemble_folders from nnUNet on the downloaded model predictions.

    This command takes the downloaded model predictions for each patient and runs
    the ensemble_folders function from nnUNet to create an ensemble prediction.

    For each patient, it runs:
    1. One ensemble for each annotation group (anno1, anno2, anno3, etc.)
    2. One ensemble on all models together

    The ensemble predictions are saved to an 'ensemble_results' folder.

    Patient selection options:
    - Custom list: Specific patient numbers (e.g., 001 034)
    """
    # Set up logger
    logger = setup_logging(verbose)

    # Create ensembling directory if it doesn't exist
    ensembling_dir = Path("ensembling")
    if not os.path.exists(ensembling_dir):
        logger.error("Ensembling directory not found. Please run the 'ensemble' command first.")
        return

    # Create output directory
    output_dir = Path("ensemble_results")
    os.makedirs(output_dir, exist_ok=True)

    # Process each patient
    for patient_id in patients:
        logger.info(f"Processing patient UKCHLL{patient_id}...")

        # Find all model folders for this patient
        patient_dir = ensembling_dir / f"UKCHLL{patient_id}"
        if not os.path.exists(patient_dir):
            logger.error(f"No data found for patient UKCHLL{patient_id}. Please run the 'ensemble' command first.")
            continue

        # Get all model folders for this patient
        model_folders = [str(d) for d in patient_dir.iterdir() if d.is_dir()]
        if not model_folders:
            logger.error(f"No model folders found for patient UKCHLL{patient_id}.")
            continue

        logger.info(f"Found {len(model_folders)} model folders for patient UKCHLL{patient_id}.")

        # Create output folder for this patient
        patient_output_dir = output_dir / f"UKCHLL{patient_id}"
        os.makedirs(patient_output_dir, exist_ok=True)

        # Group model folders by annotation
        annotation_groups = {}
        for folder in model_folders:
            # Extract the annotation number from the folder name (e.g., anno1_init112233_foldall -> 1)
            folder_name = os.path.basename(folder)
            match = re.match(r'anno(\d+)_.*', folder_name)
            if match:
                anno_num = match.group(1)
                if anno_num not in annotation_groups:
                    annotation_groups[anno_num] = []
                annotation_groups[anno_num].append(folder)

        # Run ensemble_folders for each annotation group
        for anno_num, folders in annotation_groups.items():
            if len(folders) > 1:  # Only ensemble if there's more than one model
                try:
                    # Create output folder for this annotation group
                    anno_output_dir = patient_output_dir / f"anno{anno_num}"
                    os.makedirs(anno_output_dir, exist_ok=True)

                    logger.info(f"Running ensemble_folders for patient UKCHLL{patient_id}, annotation {anno_num} with {len(folders)} models...")
                    ensemble_folders(
                        list_of_input_folders=folders,
                        output_folder=str(anno_output_dir),
                        save_merged_probabilities=True,
                        num_processes=num_processes
                    )
                    logger.info(f"Ensemble completed successfully for patient UKCHLL{patient_id}, annotation {anno_num}.")
                except Exception as e:
                    logger.error(f"Error running ensemble_folders for patient UKCHLL{patient_id}, annotation {anno_num}: {str(e)}")
                    continue
            else:
                logger.info(f"Skipping ensemble for patient UKCHLL{patient_id}, annotation {anno_num} as it only has {len(folders)} model.")

        # Run ensemble_folders for all models together
        try:
            # Create output folder for all models together
            all_models_output_dir = patient_output_dir / "all_models"
            os.makedirs(all_models_output_dir, exist_ok=True)

            logger.info(f"Running ensemble_folders for patient UKCHLL{patient_id} with all {len(model_folders)} models...")
            ensemble_folders(
                list_of_input_folders=model_folders,
                output_folder=str(all_models_output_dir),
                save_merged_probabilities=True,
                num_processes=num_processes
            )
            logger.info(f"Ensemble completed successfully for patient UKCHLL{patient_id} with all models.")
        except Exception as e:
            logger.error(f"Error running ensemble_folders for patient UKCHLL{patient_id} with all models: {str(e)}")
            continue

    logger.info(Text("All ensembling completed successfully", style="bold green"))


@app.command()
def ensemble(
    patients: List[str] = typer.Argument(..., help="List of patient numbers (e.g., 001 034) or 'all', 'train', 'validation', 'test'"),
    models: List[str] = typer.Option(["all"], "-m", "--models", help="List of models to use for ensembling (e.g., anno1_init112233_foldall) or 'all'"),
    nb_workers: int = typer.Option(10, '-j', "--jobs", help="Number of processes to run"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose logging")
) -> None:
    """
    Ensemble predictions from multiple models and upload results to S3.

    This command combines the functionality of both 'ensemble' and 'run-ensemble' commands:
    1. Downloads model predictions for each patient and model
    2. Runs ensemble_folders for each annotation group and all models
    3. Uploads the ensemble results to S3
    4. Cleans up temporary files after processing each patient

    For each patient, it runs:
    1. One ensemble for each annotation group (anno1, anno2, anno3, etc.)
    2. One ensemble on all models together

    Patient selection options:
    - 'all': All available patients
    - 'train': Predefined training set
    - 'validation': Predefined validation set
    - 'test': Predefined test set
    - Custom list: Specific patient numbers (e.g., 001 034)

    Model selection options:
    - 'all': All available models
    - Custom list: Specific model names (e.g., anno1_init112233_foldall)
    """
    # Set up logger
    logger = setup_logging(verbose)

    # Handle patient selection
    patient_selection = patients
    if len(patients) == 1:
        # Check if it's one of the predefined sets
        if patients[0] in ["all", "train", "validation", "test"]:
            patient_selection = patients[0]

    # Get selected patients
    selected_patients = get_selected_patients(patient_selection, verbose)
    if not selected_patients:
        return

    # Get available models
    available_models = get_available_models()
    if not available_models:
        logger.error("No models found in S3")
        return

    # Handle model selection
    selected_models = []
    if len(models) == 1 and models[0] == "all":
        selected_models = available_models
        info(Text(f"Selected all {len(selected_models)} models", style="bold green"))
    else:
        # Custom list of models
        missing_models = []
        for model in models:
            if model in available_models:
                selected_models.append(model)
            else:
                missing_models.append(model)

        if missing_models:
            missing_list = ", ".join(missing_models)
            info(Text.assemble(("Warning: ", "bold yellow"), (f"The following models were not found: {missing_list}", "")))

    if not selected_models:
        info(Text.assemble(("Error: ", "bold red"), ("No valid models selected", "")))
        return

    # Process each patient
    for patient_id in selected_patients:
        logger.info(f"Processing patient UKCHLL{patient_id}...")

        # Create temporary directories
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            download_dir = temp_path / "download"
            ensemble_dir = temp_path / "ensemble"

            os.makedirs(download_dir, exist_ok=True)
            os.makedirs(ensemble_dir, exist_ok=True)

            # Create patient directory
            patient_download_dir = download_dir / f"UKCHLL{patient_id}"
            os.makedirs(patient_download_dir, exist_ok=True)

            # Download model predictions for this patient
            logger.info(f"Downloading model predictions for patient UKCHLL{patient_id}...")

            # Create a list of patient-model pairs to process
            pairs = [(patient_id, model) for model in selected_models]

            # Define a function to get the combined size of all files for a patient-model pair
            def get_pair_size(pair):
                p_id, model_name = pair

                # Define the remote path
                remote_dir = f"{os.environ['S3_OUTPUT_DIR']}/UKCHLL{p_id}/{model_name}"

                # List contents of the output directory
                contents = s3_utils.list_output_directory()

                # Find all files for this patient and model
                pattern = re.compile(r'^' + re.escape(remote_dir) + r'/')
                total_size = 0

                for item in contents:
                    key = item['Key']
                    if pattern.match(key):
                        # Get the file size
                        bucket, file_key = s3_utils.parse_remote_path(key)
                        file_size = s3_utils.get_file_size(bucket, file_key)
                        total_size += file_size

                return total_size

            # Define a function to process a patient-model pair
            def process_pair(pair, progress_tracker):
                p_id, model_name = pair
                # Get the actual size of the pair
                pair_size = get_pair_size(pair)
                progress_tracker.start_file(pair, f"Processing patient UKCHLL{p_id} with model {model_name}", pair_size)
                logger.info(f"Processing patient UKCHLL{p_id} with model {model_name}...")

                try:
                    # Download CURVAS files
                    success = download_curvas_files(p_id, model_name, download_dir, verbose, progress_tracker)

                    if success:
                        progress_tracker.complete_file(f"Processed patient UKCHLL{p_id} with model {model_name}", pair_size, time.time(), success=True)
                    else:
                        progress_tracker.complete_file(f"Failed to process patient UKCHLL{p_id} with model {model_name}", pair_size, time.time(), success=False)

                except Exception as e:
                    logger.error(f"Error processing patient UKCHLL{p_id} with model {model_name}: {str(e)}")
                    progress_tracker.complete_file(f"Error processing patient UKCHLL{p_id} with model {model_name}", pair_size, time.time(), success=False)

            # Calculate the total number of files to download
            def count_pair_files(pair):
                p_id, model_name = pair

                # Define the remote path
                remote_dir = f"{os.environ['S3_OUTPUT_DIR']}/UKCHLL{p_id}/{model_name}"

                # List contents of the output directory
                contents = s3_utils.list_output_directory()

                # Find all files for this patient and model
                pattern = re.compile(r'^' + re.escape(remote_dir) + r'/')
                file_count = 0

                for item in contents:
                    key = item['Key']
                    if pattern.match(key):
                        file_count += 1

                return file_count

            # Calculate the total number of files to download
            total_files = sum(count_pair_files(pair) for pair in pairs)

            logger.info(f"This will download a total of {total_files} files for patient UKCHLL{patient_id}")

            # Track progress of processing pairs
            track_progress(pairs, get_pair_size, process_pair, total_files)

            # Get all model folders for this patient
            model_folders = [str(d) for d in patient_download_dir.iterdir() if d.is_dir()]
            if not model_folders:
                logger.error(f"No model folders found for patient UKCHLL{patient_id}.")
                continue

            logger.info(f"Found {len(model_folders)} model folders for patient UKCHLL{patient_id}.")

            # Create output folder for this patient
            patient_ensemble_dir = ensemble_dir / f"UKCHLL{patient_id}"
            os.makedirs(patient_ensemble_dir, exist_ok=True)

            # Group model folders by annotation
            annotation_groups = {}
            for folder in model_folders:
                # Extract the annotation number from the folder name (e.g., anno1_init112233_foldall -> 1)
                folder_name = os.path.basename(folder)
                match = re.match(r'anno(\d+)_.*', folder_name)
                if match:
                    anno_num = match.group(1)
                    if anno_num not in annotation_groups:
                        annotation_groups[anno_num] = []
                    annotation_groups[anno_num].append(folder)

            # Run ensemble_folders for each annotation group
            for anno_num, folders in annotation_groups.items():
                if len(folders) > 1:  # Only ensemble if there's more than one model
                    try:
                        # Create output folder for this annotation group
                        anno_output_dir = patient_ensemble_dir / f"anno{anno_num}"
                        os.makedirs(anno_output_dir, exist_ok=True)

                        logger.info(f"Running ensemble_folders for patient UKCHLL{patient_id}, annotation {anno_num} with {len(folders)} models...")
                        ensemble_folders(
                            list_of_input_folders=folders,
                            output_folder=str(anno_output_dir),
                            save_merged_probabilities=True,
                            num_processes=nb_workers
                        )
                        logger.info(f"Ensemble completed successfully for patient UKCHLL{patient_id}, annotation {anno_num}.")
                    except Exception as e:
                        logger.error(f"Error running ensemble_folders for patient UKCHLL{patient_id}, annotation {anno_num}: {str(e)}")
                        continue
                else:
                    logger.info(f"Skipping ensemble for patient UKCHLL{patient_id}, annotation {anno_num} as it only has {len(folders)} model.")

            # Run ensemble_folders for all models together
            try:
                # Create output folder for all models together
                all_models_output_dir = patient_ensemble_dir / "all_models"
                os.makedirs(all_models_output_dir, exist_ok=True)

                logger.info(f"Running ensemble_folders for patient UKCHLL{patient_id} with all {len(model_folders)} models...")
                ensemble_folders(
                    list_of_input_folders=model_folders,
                    output_folder=str(all_models_output_dir),
                    save_merged_probabilities=True,
                    num_processes=nb_workers
                )
                logger.info(f"Ensemble completed successfully for patient UKCHLL{patient_id} with all models.")
            except Exception as e:
                logger.error(f"Error running ensemble_folders for patient UKCHLL{patient_id} with all models: {str(e)}")
                continue

            # Upload ensemble results to S3
            logger.info(f"Uploading ensemble results for patient UKCHLL{patient_id} to S3...")

            # Upload each annotation group ensemble
            for anno_num in annotation_groups.keys():
                anno_output_dir = patient_ensemble_dir / f"anno{anno_num}"
                if anno_output_dir.exists():
                    # Upload to S3_OUTPUT_DIR/UKCHLL{patient_id}/ensemble_anno{anno_num}
                    upload_directory_to_s3(
                        directory=str(anno_output_dir),
                        remote_dir_env_var="S3_OUTPUT_DIR",
                        subfolder=f"UKCHLL{patient_id}/ensemble_anno{anno_num}",
                        verbose=verbose,
                        command_description=f"Upload ensemble results for patient UKCHLL{patient_id}, annotation {anno_num}",
                        tracker=False
                    )

            # Upload all models ensemble
            all_models_output_dir = patient_ensemble_dir / "all_models"
            if all_models_output_dir.exists():
                # Upload to S3_OUTPUT_DIR/UKCHLL{patient_id}/ensemble_all
                upload_directory_to_s3(
                    directory=str(all_models_output_dir),
                    remote_dir_env_var="S3_OUTPUT_DIR",
                    subfolder=f"UKCHLL{patient_id}/ensemble_all",
                    verbose=verbose,
                    command_description=f"Upload ensemble results for patient UKCHLL{patient_id}, all models",
                    tracker=False
                )

            logger.info(f"Ensemble results for patient UKCHLL{patient_id} uploaded successfully.")

        # Temporary directory is automatically cleaned up after the with block

    logger.info(Text("All ensembling and uploading completed successfully", style="bold green"))
