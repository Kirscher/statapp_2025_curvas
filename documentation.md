# StatApp Command Line Interface Documentation

## Introduction
This document provides comprehensive documentation for the StatApp command-line interface. The StatApp CLI is a medical segmentation tool that handles uncertainty and inter-annotator variability. It is built on top of nnU-Net, a ready-to-use framework for image segmentation.

The CLI provides commands for:
- Preparing datasets
- Training models
- Running predictions
- Combining multiple models
- Calculating metrics
- Managing artifacts in S3 storage

## Command Overview

### about
#### Description
Displays information about the project, including its purpose and contributors.

#### Usage
```
statapp about
```

#### Parameters
None

### upload
#### Description
The upload command provides functionality for uploading local directories to S3 storage. It includes three subcommands:

##### upload_data
Uploads a local directory to the S3 data folder defined in the .env file.

##### upload_model_artifacts
Uploads a local directory to the S3 artifacts/model folder defined in the .env file.

##### upload_preprocessing_artifacts
Uploads a local directory to the S3 artifacts/preprocessing folder defined in the .env file.

#### Usage
```
statapp upload upload-data <directory> [--verbose]
statapp upload upload-model-artifacts <directory> <modelfolder> [--verbose]
statapp upload upload-preprocessing-artifacts <directory> <preprocessingfolder> [--verbose]
```

#### Parameters
| Subcommand | Parameter | Type | Description |
|------------|-----------|------|-------------|
| upload-data | directory | string | Path of the local directory to upload |
| upload-data | --verbose, -v | flag | Enable verbose output |
| upload-model-artifacts | directory | string | Path of the local directory to upload |
| upload-model-artifacts | modelfolder | string | Name of the model subfolder |
| upload-model-artifacts | --verbose, -v | flag | Enable verbose output |
| upload-preprocessing-artifacts | directory | string | Path of the local directory to upload |
| upload-preprocessing-artifacts | preprocessingfolder | string | Name of the preprocessing subfolder |
| upload-preprocessing-artifacts | --verbose, -v | flag | Enable verbose output |

### empty-artifacts
#### Description
Removes all files and folders from the S3 artifacts folder defined in the .env file. This operation cannot be undone, so use with caution.

#### Usage
```
statapp empty-artifacts [--verbose] [--confirm]
```

#### Parameters
| Parameter | Type | Description |
|-----------|------|-------------|
| --verbose, -v | flag | Enable verbose output |
| --confirm, -c | flag | Confirm deletion without prompting |

### empty-data
#### Description
Removes all files and folders from the S3 data folder defined in the .env file. This operation cannot be undone, so use with caution.

#### Usage
```
statapp empty-data [--verbose] [--confirm]
```

#### Parameters
| Parameter | Type | Description |
|-----------|------|-------------|
| --verbose, -v | flag | Enable verbose output |
| --confirm, -c | flag | Confirm deletion without prompting |

### prepare
#### Description
The prepare command provides functionality for preparing datasets for analysis. It includes three subcommands:

##### download-dataset
Downloads a dataset for analysis without running preprocessing.

##### download-preprocessing
Downloads preprocessing artifacts for a dataset.

##### prepare
Prepares a dataset for analysis by downloading the data and running preprocessing.

#### Usage
```
statapp prepare download-dataset <annotator> <patients> [--verbose]
statapp prepare download-preprocessing <annotator> <patients> [--verbose]
statapp prepare prepare <annotator> <patients> [--skip] [--num-processes-fingerprint <num>] [--num-processes <num>] [--verbose]
```

#### Parameters
| Subcommand | Parameter | Type | Description |
|------------|-----------|------|-------------|
| download-dataset | annotator | string | Annotator (1/2/3) |
| download-dataset | patients | list/string | List of patient numbers (e.g., 001 034) or 'all', 'train', 'validation', 'test' |
| download-dataset | --verbose, -v | flag | Enable verbose logging |
| download-preprocessing | annotator | string | Annotator (1/2/3) |
| download-preprocessing | patients | list/string | List of patient numbers (e.g., 001 034) or 'all', 'train', 'validation', 'test' |
| download-preprocessing | --verbose, -v | flag | Enable verbose logging |
| prepare | annotator | string | Annotator (1/2/3) |
| prepare | patients | list/string | List of patient numbers (e.g., 001 034) or 'all', 'train', 'validation', 'test' |
| prepare | --skip | flag | Skip download and only run preprocessing |
| prepare | --num-processes-fingerprint, -npfp | integer | Number of processes to use for fingerprint extraction (default: 2) |
| prepare | --num-processes, -np | integer | Number of processes to use for preprocessing (default: 2) |
| prepare | --verbose, -v | flag | Enable verbose logging |

### train
#### Description
Runs nnUNet training. The dataset must be prepared with the prepare command beforehand.

#### Usage
```
statapp train [<seed>] [--fold <fold>] [--patients <patients>] [--annotator <annotator>] [--verbose]
```

#### Parameters
| Parameter | Type | Description |
|-----------|------|-------------|
| seed | integer | Set random seed for reproducibility |
| --fold, -f | string | Fold to use for training. Can be 'all' to use all folds, or a specific fold number (0-4) |
| --patients, -p | list/string | List of patient numbers (e.g., 001 034) or 'all', 'train', 'validation', 'test' |
| --annotator, -a | string | Annotator (1/2/3) |
| --verbose, -v | flag | Enable verbose logging |

### run
#### Description
Runs the complete pipeline: prepare data, train model, and upload artifacts. This command combines the functionality of the prepare, train, and upload_artifacts commands.

#### Usage
```
statapp run <annotator> <seed> <patients> [--fold <fold>] [--skip] [--num-processes-fingerprint <num>] [--num-processes <num>] [--verbose]
```

#### Parameters
| Parameter | Type | Description |
|-----------|------|-------------|
| annotator | string | Annotator (1/2/3) |
| seed | integer | Set random seed for reproducibility |
| patients | list/string | List of patient numbers (e.g., 001 034) or 'all', 'train', 'validation', 'test' |
| --fold, -f | string | Fold to use for training. Can be 'all' to use all folds, or a specific fold number (0-4) |
| --skip | flag | Skip download and only run preprocessing |
| --num-processes-fingerprint, -npfp | integer | Number of processes to use for fingerprint extraction (default: 2) |
| --num-processes, -np | integer | Number of processes to use for preprocessing (default: 2) |
| --verbose, -v | flag | Enable verbose logging |

### predict
#### Description
Predicts segmentation for patients using the specified models. Downloads patient images and model checkpoints, runs prediction for each model, and uploads the results to S3.

#### Usage
```
statapp predict <patients> [--models <models>] [--jobs <num>] [--verbose]
```

#### Parameters
| Parameter | Type | Description |
|-----------|------|-------------|
| patients | list/string | List of patient numbers (e.g., 001 034) or 'all', 'train', 'validation', 'test' |
| --models, -m | list/string | List of models to use for prediction (e.g., anno1_init112233_foldall) or 'all' |
| --jobs, -j | integer | Number of processes to run (default: 10) |
| --verbose, -v | flag | Enable verbose logging |

### ensemble
#### Description
The ensemble command provides functionality for combining predictions from multiple models. It includes three subcommands:

##### dl-ensemble
Downloads predictions from multiple models.

##### run-ensemble
Runs nnUNet's ensemble_folders on the downloaded model predictions.

##### ensemble
Combines the functionality of the dl-ensemble and run-ensemble commands.

#### Usage
```
statapp ensemble dl-ensemble <patients> [--models <models>] [--jobs <num>] [--verbose]
statapp ensemble run-ensemble <patients> [--verbose] [--jobs <num>]
statapp ensemble ensemble <patients> [--models <models>] [--jobs <num>] [--verbose]
```

#### Parameters
| Subcommand | Parameter | Type | Description |
|------------|-----------|------|-------------|
| dl-ensemble | patients | list/string | List of patient numbers (e.g., 001 034) or 'all', 'train', 'validation', 'test' |
| dl-ensemble | --models, -m | list/string | List of models to use for ensemble (e.g., anno1_init112233_foldall) or 'all' |
| dl-ensemble | --jobs, -j | integer | Number of processes to run (default: 10) |
| dl-ensemble | --verbose, -v | flag | Enable verbose logging |
| run-ensemble | patients | list/string | List of patient numbers (e.g., 001 034) |
| run-ensemble | --jobs, -j | integer | Number of processes to run (default: 10) |
| run-ensemble | --verbose, -v | flag | Enable verbose logging |
| ensemble | patients | list/string | List of patient numbers (e.g., 001 034) or 'all', 'train', 'validation', 'test' |
| ensemble | --models, -m | list/string | List of models to use for ensemble (e.g., anno1_init112233_foldall) or 'all' |
| ensemble | --jobs, -j | integer | Number of processes to run (default: 10) |
| ensemble | --verbose, -v | flag | Enable verbose logging |

### metrics
#### Description
The metrics command provides functionality for calculating and downloading metrics for model predictions. It includes two subcommands:

##### compute-metrics
Calculates metrics for model predictions on patient data.

##### dl-metrics
Downloads all metrics files from S3, merges them, and saves them to the working directory.

#### Usage
```
statapp metrics compute-metrics <patients> [--models <models>] [--verbose]
statapp metrics dl-metrics [--output <output_name>] [--verbose]
```

#### Parameters
| Subcommand | Parameter | Type | Description |
|------------|-----------|------|-------------|
| compute-metrics | patients | list/string | List of patient numbers (e.g., 075 034) or 'all', 'train', 'validation', 'test' |
| compute-metrics | --models, -m | list/string | List of models to use for metric calculation (e.g., anno1_init112233_foldall) or 'all' |
| compute-metrics | --verbose, -v | flag | Enable verbose logging |
| dl-metrics | --output, -o | string | Name of the output CSV file (default: metrics.csv) |
| dl-metrics | --verbose, -v | flag | Enable verbose logging |

## Examples

### Preparing a Dataset
```
# Download a dataset for annotator 1 with all patients
statapp prepare download-dataset 1 all --verbose

# Download preprocessing artifacts for annotator 1 with training patients
statapp prepare download-preprocessing 1 train --verbose

# Prepare a dataset for annotator 1 with validation patients
statapp prepare prepare 1 validation --num-processes 4 --verbose
```

### Training a Model
```
# Train a model with annotator 1, seed 42, and all folds
statapp train 42 --annotator 1 --fold all --patients train --verbose

# Run the complete pipeline for annotator 1, seed 42, and validation patients
statapp run 1 42 validation --fold all --num-processes 4 --verbose
```

### Running Predictions
```
# Predict segmentation for test patients using all models
statapp predict test --models all --jobs 8 --verbose

# Predict segmentation for specific patients using a specific model
statapp predict 001 034 --models anno1_init42_foldall --jobs 4 --verbose
```

### Combining Models
```
# Download predictions for test patients from all models
statapp ensemble dl-ensemble test --models all --jobs 8 --verbose

# Run ensemble for specific patients
statapp ensemble run-ensemble 001 034 --jobs 4 --verbose

# Run the complete ensemble pipeline for test patients
statapp ensemble ensemble test --models all --jobs 8 --verbose
```

### Calculating Metrics
```
# Calculate metrics for test patients using all models
statapp metrics compute-metrics test --models all --verbose

# Download and merge all metrics files
statapp metrics dl-metrics --output combined_metrics.csv --verbose
```

### Managing S3 Storage
```
# Upload data to S3
statapp upload upload-data ./my_data --verbose

# Upload model artifacts to S3
statapp upload upload-model-artifacts ./my_model anno1_init42_foldall --verbose

# Empty the artifacts directory (with confirmation)
statapp empty-artifacts --verbose

# Empty the data directory (without confirmation)
statapp empty-data --confirm --verbose
```

## Environment Variables
The StatApp CLI uses several environment variables for configuration. These should be defined in a .env file in the project root directory.

| Variable | Description |
|----------|-------------|
| S3_BUCKET | The name of the S3 bucket |
| S3_DATA_DIR | The directory in S3 to store data |
| S3_ARTIFACTS_DIR | The directory in S3 to store artifacts |
| S3_OUTPUT_DIR | The directory in S3 to store output |
| S3_METRICS_DIR | The directory in S3 to store metrics |
| S3_MODEL_ARTIFACTS_SUBDIR | The subdirectory for model artifacts (default: models) |
| S3_PROPROCESSING_ARTIFACTS_SUBDIR | The subdirectory for preprocessing artifacts (default: preprocessing) |