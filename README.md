# Medical Segmentation: Handling Inter-Rater Uncertainty and Variability

#### ENSAE 2025 Applied Statistics Project

Welcome to the repository for the ENSAE StatApp project! This project focuses on medical image segmentation with a special emphasis on handling inter-rater uncertainty and variability.

## Project Overview

This project is a fork of [nnU-Net v2](https://github.com/MIC-DKFZ/nnUNet), a powerful framework for medical image segmentation, with a custom command-line interface (CLI) built on top of it. The CLI provides easy access to various functionalities including dataset preparation, model training, and data management.

**Team Members:**
- Lucas CUMUNEL
- Tara LEROUX
- Léo LEROY
- Rémy SIAHAAN-GENSOLLEN

## Installation

To install and use the project:

1. Clone the repository:
   ```shell
   git clone https://github.com/your-username/statapp_2025_curvas.git
   cd statapp_2025_curvas
   ```

2. Install the package in development mode:
   ```shell
   pip install -e .
   ```

3. Set up environment variables. One way is to create a `.env` file in the project root with the following variables:
   ```dotenv
   #S3 Endpoint
   S3_ENDPOINT="https://..."

   #S3 Access key (see src/accesskey)
   S3_ACCESS_KEY="statapp-segmedic"
   #S3 Secret key (see src/accesskey)
   S3_SECRET_KEY="..."

   #S3 Bucket name, no trailing slash.
   S3_BUCKET="projet-statapp-segmedic"
   #Name of the data directory (in the S3 bucket)
   #where CT scans and annotations will be stored.
   S3_DATA_DIR="data"

   #Name of the artifacts directory (in the S3 bucket)
   #where the training outputs will be stored.
   S3_ARTIFACTS_DIR="artifacts"
   ```
## Command Line Interface

The installation makes available a command line interface with `statapp2025curvas`. To see the help and available commands, run:

```shell
statapp2025curvas --help
```

### Available Commands

#### About
Display information about the project:
```shell
statapp2025curvas about
```

#### Prepare Dataset
Download and prepare a dataset for analysis:
```shell
statapp2025curvas prepare <annotator> <patients>
```
Arguments:
- `annotator`: Annotator number (1/2/3)
- `patients`: List of patient numbers or "all" for all patients

Options:
- `--skip`: Skip download and only run preprocessing
- `--num-processes-fingerprint`: Number of processes for fingerprint extraction
- `--num-processes`: Number of processes for preprocessing
- `--verbose`: Enable verbose logging

#### Prepare Dataset (nnUNet)
Prepare a dataset using nnUNet directly:
```shell
statapp2025curvas prepare-nnunet <base_directory> <dataset>
```
Arguments:
- `base_directory`: Local directory path to the dataset
- `dataset`: ID of the dataset to preprocess

Options:
- `--raw`: Local path of raw data relative to base
- `--preprocessed`: Local path of preprocessed data relative to base

#### Train
Train a model on a given dataset:
```shell
statapp2025curvas train <base_directory>
```
Arguments:
- `base_directory`: Local directory path to the dataset

Options:
- `--preprocessed`: Local path of preprocessed data relative to base
- `--results`: Local path of results relative to base
- `--dataset`: Specific dataset to train on (optional)

#### Upload Data
Upload a local directory to the S3 data folder:
```shell
statapp2025curvas upload-data <directory>
```
Arguments:
- `directory`: Local directory path to upload

Options:
- `--verbose`: Enable verbose output

#### Upload Artifacts
Upload a local directory to the S3 artifacts folder:
```shell
statapp2025curvas upload-artifacts <directory>
```
Arguments:
- `directory`: Local directory path to upload

Options:
- `--verbose`: Enable verbose output

#### Empty Data
Remove all files and folders from the S3 data folder:
```shell
statapp2025curvas empty-data
```
Options:
- `--confirm`: Confirm deletion without prompting
- `--verbose`: Enable verbose output

#### Empty Artifacts
Remove all files and folders from the S3 artifacts folder:
```shell
statapp2025curvas empty-artifacts
```
Options:
- `--confirm`: Confirm deletion without prompting
- `--verbose`: Enable verbose output

## Project Structure

The project is organized as follows:
- `src/statapp/`: Contains the CLI implementation and utility functions
- `src/nnunetv2/`: Contains the fork of nnU-Net v2
- `src/accesskey/`: Contains access keys information for S3 integration

## Data Organization

The project follows the nnU-Net data organization:
- `nnUNet_raw/`: Contains raw datasets
- `nnUNet_preprocessed/`: Contains preprocessed data
- `nnUNet_results/`: Contains training results

## License

This project is based on nnU-Net, which is licensed under the Apache License 2.0.
