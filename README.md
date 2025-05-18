# Reducing Uncertainty in Medical Segmentation with Ensemble Methods

## ENSAE 2025 Applied Statistics Project

This repository contains the implementation of the ENSAE StatApp project focused on reducing uncertainty in medical image segmentation through ensemble methods. The project builds upon the [nnU-Net v2](https://github.com/MIC-DKFZ/nnUNet) framework with custom modifications to handle inter-rater variability and uncertainty quantification.

## Project Overview

Medical image segmentation, particularly the automatic delineation of organs and structures, is crucial for diagnosis, treatment planning, and clinical monitoring. However, segmentation tasks often face uncertainty due to:

1. **Aleatoric uncertainty**: Inherent ambiguities in the data, including inter-rater variability (different experts annotating the same image differently)
2. **Epistemic uncertainty**: Model-related uncertainty due to limited knowledge or training data

This project systematically evaluates ensemble methods for U-Net models to reduce these uncertainties in medical segmentation tasks. We train and infer multiple models on CT scans annotated by different experts, combine them using ensemble methods, and evaluate their accuracy, aleatoric uncertainty, and epistemic uncertainty.

### Team Members
- Lucas CUMUNEL
- Tara LEROUX
- Léo LEROY
- Rémy SIAHAAN-GENSOLLEN

## Features

### Key Modifications to nnU-Net

1. **Early Stopping**
   - Added patience-based early stopping (default: 20 epochs)
   - Limited maximum training to 300 epochs
   - Prevents overfitting and reduces training time

2. **Seed Initialization**
   - Added reproducible weight initialization with fixed random seeds
   - Allows exploring different points in the loss landscape
   - Implemented through the `SeededInitWeights_He` class

3. **Enhanced Logging**
   - Improved console output with rich formatting and emojis
   - Better visualization of training metrics
   - Detailed progress tracking

4. **S3 Integration**
   - Added support for storing and retrieving data from S3-compatible storage
   - Facilitates distributed training and large dataset handling

5. **Ensemble Methods**
   - Implemented model ensembling to reduce uncertainty
   - Support for different ensemble strategies (per-annotator and global)

6. **Uncertainty Quantification**
   - Comprehensive evaluation framework for uncertainty metrics
   - Integration with CURVAS and ValUES frameworks for uncertainty assessment

### Evaluation Metrics

The project implements various metrics to evaluate model performance and uncertainty:

1. **Performance Metrics**
   - Consensus-based DICE
   - Continuous Ranked Probability Score (CRPS)
   - Hausdorff Distance

2. **Aleatoric Uncertainty Metrics**
   - Confidence (Uncertainty Assessment)
   - Normalized Cross Correlation (NCC)

3. **Epistemic Uncertainty Metrics**
   - Expected Calibration Error (ECE)
   - Average Calibration Error (ACE)
   - Area Under the Receiver Operating Characteristic curve (AUROC)
   - Area Under the Risk Curve (AURC)
   - Expected Area Under the Risk Curve (EAURC)

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

3. Set up environment variables by creating a `.env` file in the project root:
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

   #Name of the output directory (in the S3 bucket)
   S3_OUTPUT_DIR="output"

   #Name of the metrics directory (in the S3 bucket)
   S3_METRICS_DIR="metrics"

   #Subdirectories for artifacts
   S3_MODEL_ARTIFACTS_SUBDIR="models"
   S3_PROPROCESSING_ARTIFACTS_SUBDIR="preprocessing"
   ```
## Usage

The installation makes available a command-line interface with `statapp`. To see the help and available commands, run:

```shell
statapp --help
```

For detailed documentation of all commands and their parameters, see the [CLI Documentation](documentation.md).

### Basic Workflow

1. **Prepare Data**
   ```shell
   statapp prepare prepare 1 train --num-processes 4 --verbose
   ```

2. **Train Model**
   ```shell
   statapp train 112233 --annotator 1 --fold all --patients train --verbose
   ```

3. **Run Predictions**
   ```shell
   statapp predict test --models all --jobs 8 --verbose
   ```

4. **Create Ensemble**
   ```shell
   statapp ensemble ensemble test --models all --jobs 8 --verbose
   ```

5. **Calculate Metrics**
   ```shell
   statapp metrics compute-metrics test --models all --verbose
   ```

6. **Download and Combine Metrics**
   ```shell
   statapp metrics dl-metrics --output combined_metrics.csv --verbose
   ```

## Project Structure

The project is organized as follows:

- `src/statapp/`: Contains the CLI implementation and utility functions
- `src/nnunetv2/`: Contains the modified nnU-Net v2 code
  - `training/nnUNetTrainer/variants/nnUNetTrainer_Statapp.py`: Custom trainer with early stopping and seed initialization
- `src/accesskey/`: Contains access keys information for S3 integration
- `documentation.md`: Documentation of the CLI

In French : 
- `documentation-md.md`: Documentation de la CLI
- `rapport.pdf`: Rapport du projet détaillé
- `note-de-synthese.pdf`: Note de synthèse

## Data Organization

The project follows the nnU-Net data organization:
- `nnUNet_raw/`: Contains raw datasets
- `nnUNet_preprocessed/`: Contains preprocessed data
- `nnUNet_results/`: Contains training results

## Results

Our results indicate that ensemble methods significantly reduce uncertainty in medical segmentation without degrading prediction accuracy. The global ensemble model (combining all 9 individual models) showed the best performance in terms of uncertainty reduction, with statistically significant improvements in Expected Calibration Error (ECE).

For detailed results and analysis, please refer to the project report (`rapport.pdf`) and summary note (`note-de-synthese.pdf`), both in French.

## License

This project is based on nnU-Net, which is licensed under the Apache License 2.0.
