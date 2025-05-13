"""
This code aims at computing metrics for nnUNet segmentation predictions. It has 3 parts : 
-Imports
-Functions (metric computation and preprocess functions)
-Body (applying the functions)

It requires a folder with for each patient a folder containing GT and predictions (see BODY part for the detail)
"""

#IMPORTS
"""
The needed imports for the metrics computation.
Please run : 
pip install scikit-learn torch scipy monai torchmetrics
to install the libraries that are not automatically implemented by onyxia.
"""

import os
import re
import numpy as np
import pandas as pd
from sklearn.utils import column_or_1d
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score
import torch
from scipy.stats import norm
from monai.metrics import DiceMetric
from monai.transforms import AsDiscrete
from monai.transforms import CropForeground
from scipy.integrate import quad
from scipy.interpolate import interp1d
from scipy.stats import entropy
from scipy.spatial.distance import directed_hausdorff
from torchmetrics.classification import MulticlassCalibrationError
import nibabel as nib

#FUNCTIONS

"""
Metrics computation functions :
- consensus_dice_score (DICE)
- expected_calibration_error, multirater_expected_calibration_error (ECE)
- calc_ace, calib_stats, prepare_inputs_for_ace (ACE)
- volume_metric, compute_probabilistic_volume, calculate_volumes_distributions, heaviside, crps_computation (CRPS)

"""

'''
Dice Score Evaluation
'''

def consensus_dice_score(groundtruth, bin_pred, prob_pred):
    """
    Computes an average of dice score for consensus areas only.
    
    groundtruth: numpy stack list containing the three ground truths [gt1, gt2, gt3]
                 each gt has the following values: 1: pancreas, 2: kidney, 3: liver
                    (3, slices, X, Y)
    bin_pred: binarized prediction matrix containing values: {0,1,2,3}
    prob_pred: probability prediction matrix, shape: (3, slices, X, Y), the three being
                a probability matrix per each class
     
    @output dice_scores, confidence
  
    """

    # Transform probability predictions to one-hot encoding by taking the argmax
    prediction_onehot = AsDiscrete(to_onehot=4)(torch.from_numpy(np.expand_dims(bin_pred, axis=0)))[1:].astype(np.uint8)
    
    # Split ground truth into separate organs and calculate consensus
    organs =  {1: 'panc', 2: 'kidn', 3: 'livr'}
    consensus = {}
    dissensus = {}

    for organ_val, organ_name in organs.items():
        # Get the ground truth for the current organ
        organ_gt = (groundtruth == organ_val).astype(np.uint8)
        organ_bck = (groundtruth != organ_val).astype(np.uint8)
        
        # Calculate consensus regions (all annotators agree)
        consensus[organ_name] = np.logical_and.reduce(organ_gt, axis=0).astype(np.uint8)
        consensus[f"{organ_name}_bck"] = np.logical_and.reduce(organ_bck, axis=0).astype(np.uint8)
        
        # Calculate dissensus regions (where both background and foreground are 0)
        dissensus[organ_name] = np.logical_and(consensus[organ_name] == 0, 
                                               consensus[f"{organ_name}_bck"]== 0).astype(np.uint8)
    
    # Mask the predictions and ground truth with the consensus areas
    predictions = {}
    groundtruth_consensus = {}
    confidence = {}

    for organ_val, organ_name in organs.items():
        # Apply the dissensus mask to exclude non-consensus areas
        filtered_prediction = prediction_onehot[organ_val-1] * (1 - dissensus[organ_name])
        filtered_groundtruth = consensus[organ_name] * (1 - dissensus[organ_name])
        
        predictions[organ_name] = filtered_prediction
        groundtruth_consensus[organ_name] = filtered_groundtruth
        
        # Compute mean probabilities and confidence in the consensus area
        prob_in_consensus_organ = prob_pred[organ_val-1] * np.where(consensus[organ_name]==1, 1, np.nan)
        prob_in_consensus_bck = prob_pred[organ_val-1] * np.where(consensus[f"{organ_name}_bck"]==1, 1, np.nan)
        mean_conf_organ = np.nanmean(prob_in_consensus_organ)
        mean_conf_bck = np.nanmean(prob_in_consensus_bck)        
        confidence[organ_name] = (((1-mean_conf_bck)+mean_conf_organ)/2)
    
    # Create DiceMetric instance
    dice_metric = DiceMetric(include_background=False, reduction="mean", get_not_nans=False, ignore_empty=True)

    dice_scores = {}
    for organ_name in organs.values():
        gt = torch.from_numpy(groundtruth_consensus[organ_name])
        pred = torch.from_numpy(predictions[organ_name])
        dice_metric.reset()
        dice_metric(pred, gt)
        dice_scores[organ_name] = dice_metric.aggregate().item()
    
    return dice_scores, confidence


'''
Entropy
'''


def calculate_entropy(image):
    """
    Calculate the entropy of an image (prediction or GT).
    
    Parameters:
    - image: numpy array representing the image or volume (slices, X, Y)
    """
    # Flatten the image and compute the histogram
    image_flat = image.flatten()
    
    # Calculate the histogram and normalize it
    hist, _ = np.histogram(image_flat, bins=256, range=(0, 256), density=True)
    
    # Compute the Shannon entropy
    entropy_value = entropy(hist)
    
    return entropy_value

'''
Hausdorff distance
'''

def calculate_hausdorff_distances (groundtruth, pred):
    """
    Calculate the Hausdorff distance between the ground truth and prediction.
    
    Parameters:
    - gt: Ground truth binary segmentation 
    - pred: Predicted binary segmentation 
    
    @output hausdorff_distance
    """
    hausdorff_distances = {}
    organs = ['panc', 'kidn', 'livr']
    
    for i, organ in enumerate(organs, start=0):

        # Extract the coordinates of the foreground (non-zero) pixels
        gt_coords = np.column_stack(np.where(groundtruth[i] > 0))
        pred_coords = np.column_stack(np.where(pred > 0))
    
        # Calculate the directed Hausdorff distance in both directions
        forward_hausdorff = directed_hausdorff(gt_coords, pred_coords)[0]
        backward_hausdorff = directed_hausdorff(pred_coords, gt_coords)[0]
    
        # The Hausdorff distance is the maximum of the two directed distances
        hausdorff_distance = max(forward_hausdorff, backward_hausdorff)
        hausdorff_distances[organ] = hausdorff_distance

    
    return hausdorff_distances

'''
AUROC
'''
def compute_auroc(groundtruth, prob_pred):
    """
    Computes the AUROC (Area Under the Receiver Operating Characteristic Curve)
    for each class in a multi-class segmentation task.

    groundtruth: numpy array, shape (classes, slices, X, Y)
                 Ground truth segmentation masks (binary, 1 for presence of class, 0 for absence).
                 This should have dimensions: [3, slices, X, Y] where 3 is the number of classes (pancreas, kidney, liver).

    prob_pred: numpy array, shape (classes, slices, X, Y)
               The predicted probabilities for each class (output of softmax or sigmoid function).

    @output  auroc_dict
    """

    # Reshape groundtruth and prob_pred into 2D arrays for each class
    auroc_dict = {}
    organs =  {1: 'panc', 2: 'kidn', 3: 'livr'}
    num_classes = groundtruth.shape[0]  # Number of classes (3 for pancreas, kidney, liver)
    
    for i in range(num_classes):
        # Flatten the groundtruth for class i and the predicted probabilities for class i
        gt_class = (groundtruth[i] > 0).astype(np.uint8).flatten()  # Convert to binary groundtruth
        prob_class = prob_pred[i].flatten()  # Flatten the probabilities for class i
        
        # Calculate AUROC score for class i
        auroc_score = roc_auc_score(gt_class, prob_class)
        auroc_dict[organs[i + 1]] = auroc_score

    return auroc_dict

'''
AURC and EAURC
'''

def rc_curve_stats(risks: np.array, confids: np.array) -> tuple[list[float], list[float], list[float]]:
    coverages = []
    selective_risks = []
    assert len(risks.shape) == 1 and len(confids.shape) == 1 and len(risks) == len(confids)

    n_samples = len(risks)
    idx_sorted = np.argsort(confids)

    coverage = n_samples
    error_sum = sum(risks[idx_sorted])

    coverages.append(coverage / n_samples)
    selective_risks.append(error_sum / n_samples)

    weights = []
    tmp_weight = 0
    for i in range(0, len(idx_sorted) - 1):
        coverage = coverage - 1
        error_sum = error_sum - risks[idx_sorted[i]]
        tmp_weight += 1
        if i == 0 or confids[idx_sorted[i]] != confids[idx_sorted[i - 1]]:
            coverages.append(coverage / n_samples)
            selective_risks.append(error_sum / (n_samples - 1 - i))
            weights.append(tmp_weight / n_samples)
            tmp_weight = 0
    return coverages, selective_risks, weights


def aurc(risks: np.array, confids: np.array):
    _, risks, weights = rc_curve_stats(risks, confids)
    return sum(
        [(risks[i] + risks[i + 1]) * 0.5 * weights[i] for i in range(len(weights))]
    )


def eaurc(risks: np.array, confids: np.array):
    """Compute normalized AURC, i.e. subtract AURC of optimal CSF (given fixed risks)."""
    n = len(risks)
    # optimal confidence sorts risk. Asencding here because we start from coverage 1/n
    selective_risks = np.sort(risks).cumsum() / np.arange(1, n + 1)
    aurc_opt = selective_risks.sum() / n
    return aurc(risks, confids) - aurc_opt


def compute_aurc_eaurc (groundtruth, prob_pred):
    """
    Compute AURC and EAURC for a single class in the segmentation task.
    
    groundtruth: numpy array, shape (slices, X, Y)
                 Binary groundtruth segmentation mask for a specific class.
    
    prob_pred: numpy array, shape (num_classes, slices, X, Y)
               Predicted probabilities for each class (output of softmax/sigmoid).
    
    @output aurc_scores, eaurc_scores
    """
    aurc_scores = {}
    eaurc_scores = {}
    organs = ['panc', 'kidn', 'livr']

    for i, organ in enumerate(organs):
        
        # Flatten groundtruth and probabilities for the specific class
        gt_class = (groundtruth[i] > 0).astype(np.uint8).flatten()
        prob_class = prob_pred[i].flatten()

        # Compute risk (error) for each pixel (1 if incorrect, 0 if correct)
        risks = np.abs(gt_class - prob_class)  # Binary error (0 or 1)
        confids = prob_class  # Confidence is the predicted probability for the class

        # Calculate AURC and EAURC for this class
        class_aurc = aurc(risks, confids)
        class_eaurc = eaurc(risks, confids)
        aurc_scores[organ] = class_aurc
        eaurc_scores[organ] = class_eaurc
    
    return aurc_scores, eaurc_scores

'''
Expected Calibration Error
'''

def multirater_ece (annotations_list, prob_pred):
    """
    Returns a list of length three of the Expected Calibration Error (ECE) per annotation.
    
    annotations_list: list of length three containing the three annotations
    prob_pred: probability prediction matrix, shape: (3, slices, X, Y), the three being
                a probability matrix per each class
     
    @output ece_dict
    """
    
    ece_dict = {1: 0, 2: 0, 3: 0}

    for i in range(3):
        ece_dict[i+1] = expected_calibration_error(annotations_list[i], prob_pred)
        
    return ece_dict


def expected_calibration_error(groundtruth, prob_pred_onehot, num_classes=4, n_bins=50):
    """
    Computes the Expected Calibration Error (ECE) between the given annotation and the 
    probabilistic prediction
    
    groundtruth: groundtruth matrix containing the following values: 1: pancreas, 2: kidney, 3: liver
                    shape: (slices, X, Y)
    prob_pred_onehot: probability prediction matrix, shape: (3, slices, X, Y), the three being
                    a probability matrix per each class
    num_classes: number of classes
    n_bins: number of bins                    
                    
    @output ece
    """ 
    
    # Convert inputs to torch tensors
    all_groundtruth = torch.tensor(groundtruth)
    all_samples = torch.tensor(prob_pred_onehot)
    
    # Calculate the probability for the background class
    background_prob = 1 - all_samples.sum(dim=0, keepdim=True)
    
    # Combine background probabilities with the provided probabilities
    all_samples_with_bg = torch.cat((background_prob, all_samples), dim=0)
    
    # Flatten the tensors to (num_samples, num_classes) and (num_samples,)
    all_groundtruth_flat = all_groundtruth.reshape(-1)
    all_samples_flat = all_samples_with_bg.permute(1, 2, 3, 0).reshape(-1, num_classes)
    
    # Initialize the calibration error metric
    calibration_error = MulticlassCalibrationError(num_classes=num_classes, n_bins=n_bins)

    # Calculate the ECE
    ece = calibration_error(all_samples_flat, all_groundtruth_flat).cpu().detach().numpy().astype(np.float64)
    
    return ece


'''
Average Calibration Error
'''

def prepare_inputs_for_ace(groundtruth, bin_pred, prob_pred):
    background_prob = 1 - np.sum(prob_pred, axis=0, keepdims=True)
    prob_pred_full = np.concatenate([background_prob, prob_pred], axis=0)

    confids = np.max(prob_pred_full, axis=0)
    flat_pred = bin_pred.flatten()
    flat_gt = groundtruth.flatten()
    flat_conf = confids.flatten()

    correct = (flat_pred == flat_gt).astype(np.int32)

    return correct, flat_conf

def calib_stats(correct, calib_confids):
    n_bins = 20
    y_true = column_or_1d(correct)
    y_prob = column_or_1d(calib_confids)

    if y_prob.min() < 0 or y_prob.max() > 1:
        raise ValueError("y_prob has values outside [0, 1]")

    labels = np.unique(y_true)
    if len(labels) > 2:
        raise ValueError(f"Only binary classification is supported. Provided labels {labels}.")
    y_true = label_binarize(y_true, classes=labels)[:, 0]

    bins = np.linspace(0.0, 1.0 + 1e-8, n_bins + 1)
    binids = np.digitize(y_prob, bins) - 1

    bin_sums = np.bincount(binids, weights=y_prob, minlength=len(bins))
    bin_true = np.bincount(binids, weights=y_true, minlength=len(bins))
    bin_total = np.bincount(binids, minlength=len(bins))

    nonzero = bin_total != 0
    num_nonzero = len(nonzero[nonzero == True])
    prob_true = bin_true[nonzero] / bin_total[nonzero]
    prob_pred = bin_sums[nonzero] / bin_total[nonzero]

    bin_discrepancies = np.abs(prob_true - prob_pred)
    return bin_discrepancies, num_nonzero

def multirater_ace(annotations, bin_pred, prob_pred):

    ace_dict = {1: 0, 2: 0, 3: 0}
    for i in range(3):
        gt_i = annotations[i]
        correct, calib_confids = prepare_inputs_for_ace(gt_i, bin_pred, prob_pred)
        bin_discrepancies, num_nonzero = calib_stats(correct, calib_confids)
        ace_score = (1 / num_nonzero) * np.sum(bin_discrepancies)
        ace_dict[i+1] = ace_score
    return ace_dict

'''
CRPS  evaluation
'''

def volume_metric(groundtruth, prediction, voxel_proportion=1):
    """
    Calculates the Continuous Ranked Probability Score (CRPS) for each volume class,
    by using the ground truths to create a probabilistic distribution that keeps the
    multirater information of having multiple annotations. 
    
    groundtruth: numpy stack list containing the three ground truths [gt1, gt2, gt3]
                 each gt has the following values: 1: pancreas, 2: kidney, 3: liver
                    (3, slices, X, Y)
    prob_pred: probability prediction matrix, shape: (3, slices, X, Y), the three being
                a probability matrix per each class
    voxel_proportion: vaue of the resampling needed voxel-wise, 1 by default
     
    @output crps_dict
    """
    
    cdf_list = calculate_volumes_distributions(groundtruth, voxel_proportion)
        
    crps_dict = {}    
    organs =  {1: 'panc', 2: 'kidn', 3: 'livr'}

    for organ_val, organ_name in organs.items():
        probabilistic_volume = compute_probabilistic_volume(prediction[organ_val-1], voxel_proportion)
        crps_dict[organ_name] = crps_computation(probabilistic_volume, cdf_list[organ_name], mean_gauss[organ_name], var_gauss[organ_name])

    return crps_dict


def heaviside(x):
    return 0.5 * (np.sign(x) + 1)


def crps_computation(predicted_volume, cdf, mean, std_dev):
    """
    Calculates the Continuous Ranked Probability Score (CRPS) for each volume class,
    by using the ground truths to create a probabilistic distribution that keeps the
    multirater information of having multiple annotations. 
    
    predicted_volume: scalar value representing the volume obtained from the 
                        probabilistic prediction
    cdf: cumulative density distribution (CDF) of the groundtruth volumes
    mean: mean of the gaussian distribution obtained from the three groundtruth volumes
    std_dev: std_dev of the gaussian distribution obtained from the three groundtruth volumes
     
    @output crps_dict
    """
    
    def integrand(y):
        return (cdf(y) - heaviside(y - predicted_volume)) ** 2
    
    lower_limit = mean - 3 * std_dev
    upper_limit = mean + 3 * std_dev
    
    crps_value, _ = quad(integrand, lower_limit, upper_limit) #augmenter la limite pour de meilleurs résultats ?
        
    return crps_value


def calculate_volumes_distributions(groundtruth, voxel_proportion=1):
    """
    Calculates the Cumulative Distribution Function (CDF) of the Probabilistic Function Distribution (PDF)
    obtained by calcuating the mean and the variance of considering the three annotations.
    
    groundtruth: numpy stack list containing the three ground truths [gt1, gt2, gt3]
                 each gt has the following values: 1: pancreas, 2: kidney, 3: liver
                    (3, slices, X, Y)
    voxel_proportion: vaue of the resampling needed voxel-wise, 1 by default
    
    @output cdfs_dict
    """
    
    organs = {1: 'panc', 2: 'kidn', 3: 'livr'}
    
    global mean_gauss, var_gauss, volumes  # Make them global to access in crps
    mean_gauss = {}
    var_gauss = {}
    volumes = {}

    for organ_val, organ_name in organs.items():
        volumes[organ_name] = [np.unique(gt, return_counts=True)[1][organ_val] * np.prod(voxel_proportion) for gt in groundtruth]
        mean_gauss[organ_name] = np.mean(volumes[organ_name])
        var_gauss[organ_name] = np.std(volumes[organ_name])

    # Create normal distribution objects
    gaussian_dists = {organ_name: norm(loc=mean_gauss[organ_name], scale=var_gauss[organ_name]) for organ_name in organs.values()}
    
    # Generate CDFs
    cdfs = {}
    for organ_name in organs.values():
        x = np.linspace(gaussian_dists[organ_name].ppf(0.01), gaussian_dists[organ_name].ppf(0.99), 100)
        cdf_values = gaussian_dists[organ_name].cdf(x)
        cdfs[organ_name] = interp1d(x, cdf_values, bounds_error=False, fill_value=(0, 1))  # Create an interpolation function

    return cdfs
    
    
def compute_probabilistic_volume(preds, voxel_proportion=1):
    """
    Computes the volume of the matrix given (either pancreas, kidney or liver)
    by adding up all the probabilities in this matrix. This way the uncertainty plays
    a role in the computation of the predicted organ. If there is no uncertainty, the 
    volume should be close to the mean obtained by averaging the three annotations.
    
    preds: probabilistic matrix of a specific organ
    voxel_proportion: vaue of the resampling needed voxel-wise, 1 by default
     
    @output volume
    """
    
    # Sum the predicted probabilities to get the volume
    volume = preds.sum().item()
    
    return volume*voxel_proportion

'''
NCC
'''

def compute_ncc(annotations, prob_pred):
    """
    Compute the normalized cross correlation between a ground truth uncertainty and a predicted uncertainty map,
    to determine how similar the maps are.
    :param gt_unc_map: the ground truth uncertainty map based on the rater variability
    :param pred_unc_map: the predicted uncertainty map
    :return: float: the normalized cross correlation between gt and predicted uncertainty map
    """

    ncc_dict = {}
    organ_name = ['panc', 'kidn', 'livr']

    # Uncertainty from GT (std across raters)
    gt_masks = np.stack(annotations, axis=0)
    gt_unc_map = np.std(gt_masks.astype(np.float32), axis=0)

    # Uncertainty from prediction (entropy)
    class_probs = np.array(prob_pred)  # [C, D, H, W]
    entropy_map = -np.sum(class_probs * np.log(class_probs + 1e-8), axis=0)

    # NCC
    for i, organ in enumerate(organ_name):
        
        mu_gt = np.mean(gt_unc_map)
        mu_pred = np.mean(entropy_map)
        sigma_gt = np.std(gt_unc_map, ddof=1)
        sigma_pred = np.std(entropy_map, ddof=1)
        gt_norm = gt_unc_map - mu_gt
        pred_norm = entropy_map - mu_pred
        prod = np.sum(np.multiply(gt_norm, pred_norm))
        ncc = (1 / (np.size(gt_unc_map) * sigma_gt * sigma_pred)) * prod
        ncc_dict[organ] = ncc
    return ncc_dict

"""
Preprocessing functions : 
- files_to_data (extracts and converts data)
- preprocess_results (cropping)
"""

def preprocess_results(ct_image, annotations, results):
    """
    Preprocess the images, predictions and annotations in order to be evaluated.
    It crops the foreground and applies the same crop to the rest of matrices.
    This is done to save some memory and work with smaller matrices.
    
    ct_image: CT images of shape (slices, X, Y)
    annotations: list containing the three ground truths [gt1, gt2, gt3]
                 each gt has the following values: 1: pancreas, 2: kidney, 3: liver
                 each gt has the following shape (slices, X, Y)
    results: list containing the results [binarized prediction, 
                                          pancreas_confidence,
                                          kidney_confidence,
                                          liver_confidence
                                         ]
            the binarized prediction the following values: 1: pancreas, 2: kidney, 3: liver
            each confidence has probabilistic values that range from 0 to 1
     
    @output cropped_annotations, cropped_results[0], cropped_results[1:]
  
    """

    # Define the CropForeground transform
    cropper = CropForeground(select_fn=lambda x: x > 0)  # Assuming non-zero voxels are foreground

    # Compute the cropping box based on the CT image
    box_start, box_end = cropper.compute_bounding_box(ct_image)
    
    # Apply the cropping box to all annotations
    cropped_annotations = [annotation[..., box_start[0]:box_end[0], box_start[1]:box_end[1]] for annotation in annotations]
    cropped_results = [result[..., box_start[0]:box_end[0], box_start[1]:box_end[1]] for result in results]

    return cropped_annotations, cropped_results[0], cropped_results[1:]


def files_to_data (result_file, prob_file, gt_folder) :
    """ 
    From patient files (paths) to data (arrays).
    result_file : the result.nii.gz containing the predicted segmentation
    prob_file : the result.npz containing the probabilities for each class
    patient_folder : the folder containing the raw image and the 3 annotations
    """
    #Finding the names
    gt1_file, gt2_file, gt3_file = [f for f in os.listdir(gt_folder) if re.findall(r"annotation", f)]
    ct_file = [f for f in os.listdir(gt_folder) if re.findall(r"image", f)]

    #Loading the files for GT with (slices, X, Y) shape
    gt1 = nib.load(gt_folder+"/"+gt1_file).get_fdata()
    gt1 = gt1.transpose(2, 0, 1)
    
    gt2 = nib.load(gt_folder+"/"+gt2_file).get_fdata()
    gt2 = gt2.transpose(2, 0, 1)
    
    gt3 = nib.load(gt_folder+"/"+gt3_file).get_fdata()
    gt3 = gt3.transpose(2, 0, 1)

    annotations = [gt1, gt2, gt3]
    #Loading the raw file with (slices, X, Y) shape
    ct_image = nib.load(gt_folder+"/"+ct_file[0]).get_fdata()
    ct_image = ct_image.transpose(2, 0, 1)

    #Loading the prediction file with (slices, X, Y) shape
    bin_pred = nib.load(result_file).get_fdata().astype(np.uint8) 
    bin_pred = bin_pred.transpose(2, 1, 0)

    #Loading the raw file with (classes, slices, X, Y) shape
    prob_data = np.load(prob_file)
    prob_data = prob_data[prob_data.files[0]]

    #Extracting the probabilites per class (pancreas, kidney and liver)
    pancreas_conf = prob_data[1].transpose(0, 2, 1)
    kidney_conf = prob_data[2].transpose(0, 2, 1)
    liver_conf = prob_data[3].transpose(0, 2, 1)

    results = [bin_pred, pancreas_conf, kidney_conf, liver_conf]

    return ct_image, annotations, results

'''
Applying the metrics to the inputed data
'''

def apply_metrics (l_patient_files):
    '''
    Apply all the metrics.
    l_patient_files : dict of file paths : {"pred" : path_to/result.nii.gz", "prob" : "path_to/result.npz" (containing the probabilities), "gt" : "path_to/GT_patient folder" (containing the 3 annotations and the raw image)}

    '''
    #extracting patient ID
    ct_name=re.findall(r"\/([^\/]+)$",l_patient_files["gt"])

    #from files to data
    ct_image, annotations, results = files_to_data(l_patient_files["pred"], l_patient_files["prob"], l_patient_files["gt"])

    #preprocess the data
    cropped_annotations, cropped_bin_pred, cropped_prob_pred = preprocess_results(ct_image, annotations, results)

    #DICE
    print( "Computing DICE")
    #dice_scores, confidence = consensus_dice_score(np.stack(cropped_annotations, axis=0), cropped_bin_pred, cropped_prob_pred)
    #print(f"DICE : {dice_scores}")

    #GT Entropy
    print("Computing Entropies")
    #entropy_gt = calculate_entropy(np.stack(cropped_annotations, axis=0))
    #Prediction Entropy
    #entropy_pred = calculate_entropy(cropped_bin_pred)
    #print(f"Entropy GT: {entropy_gt}, Entropy Pred: {entropy_pred}")

    #Hausdorff Distance
    print("Computing Hausdorff Distance")
    #hausdorff_distances=calculate_hausdorff_distances(cropped_annotations,cropped_bin_pred)
    #print(f"Hausdorff Distances: {hausdorff_distances}")
    
    #AUROC
    print("Computing AUROC")
    #auroc_scores = compute_auroc(np.stack(cropped_annotations, axis=0), cropped_prob_pred)
    #print(f"AUROC: {auroc_scores}")

    #AURC and EAURC
    print("Computing AURC and EAURC")
    #aurc_scores, eaurc_scores = compute_aurc_eaurc(np.stack(cropped_annotations, axis=0), cropped_prob_pred)
    #print(f"AURC: {aurc_scores}")
    #print(f"EAURC: {eaurc_scores}")
    

    #ECE
    print("Computing ECE")
    #ece_scores = multirater_expected_calibration_error(cropped_annotations, cropped_prob_pred)
    #print(f"ECE : {ece_scores}")

    #ACE
    print("Computing ACE")
    #ace_dict=multirater_ace(cropped_annotations, cropped_bin_pred, cropped_prob_pred)
    #print(f"ACE : {ace_dict}")
    

    #CRPS
    print("Computing CRPS")
    #crps_score = volume_metric(np.stack(cropped_annotations, axis=0), cropped_prob_pred)
    #print(f"CRPS : {crps_score}")

    #NCC
    print("Computing NCC")
    ncc_dict = compute_ncc(cropped_annotations,cropped_prob_pred)
    print(f"NCC : {ncc_dict}")



    return {"CT" : ct_name, "DICE_panc" : dice_scores['panc'], "DICE_kidn" : dice_scores['kidn'], "DICE_livr" : dice_scores['livr'], "Entropy_GT" : entropy_gt, "Entropy_Pred" : entropy_pred, "Hausdorff_panc" : hausdorff_distances['panc'], "Hausdorff_kidn" : hausdorff_distances['kidn'], "Hausdorff_livr" : hausdorff_distances['livr'], "AUROC_panc" : auroc_scores["panc"], "AUROC_kidn" : auroc_scores["kidn"], "AUROC_livr" : auroc_scores["livr"], "AURC_panc" : aurc_scores["panc"], "AURC_kidn" : aurc_scores["kidn"], "AURC_livr" : aurc_scores["livr"], "EAURC_panc": eaurc_scores["panc"], "EAURC_kidn" : eaurc_scores["kidn"], "EAURC_livr" : eaurc_scores["livr"], "ECE_0" : ece_scores[0], "ECE_1" : ece_scores[1], "ECE_2" : ece_scores[2], "ACE_0" : ace_dict[0], "ACE_1" : ace_dict[1], "ACE_2" : ace_dict[2], "CRPS_panc" : crps_score['panc'], "CRPS_kidn" : crps_score['kidn'], "CRPS_livr" : crps_score['livr'], "NCC" : ncc_score}



#BODY
"""
Gather the input locations and apply the metrics to all predictions.
The input folder should be looking like this : 
-- Folder   -- Patient_01   -- pred_01.nii.gz
                            -- pred_prob_01.npz
                            -- GT_01                -- image.nii.gz
                                                    -- annotation_1.nii.gz
                                                    -- annotation_2.nii.gz
                                                    -- annotation_3.nii.gz
            -- Patient_02   -- pred_02.nii.gz
            ...
            ...
            ...
"""


#Locating data
data_path = "/home/onyxia/statapp_2025_curvas/truc"#input(str("Path to the folder with all the patient folders : "))
l_patients_path = os.listdir(data_path)
l_patients=[]

for patient_dir in l_patients_path:
    patient_dict = {"pred": None, "prob": None, "gt": None}
    patient_path = os.path.join(data_path, patient_dir)
    for item in os.listdir(patient_path):
            item_path = os.path.join(patient_path, item)
            if item.endswith(".nii.gz"):
                patient_dict["pred"] = item_path
            elif item.endswith(".npz"):
                patient_dict["prob"] = item_path
            elif os.path.isdir(item_path):
                patient_dict["gt"] = item_path

    l_patients.append(patient_dict)

#Prepare output
df = pd.DataFrame(columns=["CT", "DICE_panc", "DICE_kidn", "DICE_livr", "Entropy_GT", "Entropy_Pred", "Hausdorff_panc", "Hausdorff_kidn", "Hausdorff_livr", "AUROC_panc", "AUROC_kidn", "AUROC_livr", "AURC_panc", "AURC_kidn", "AURC_livr", "EAURC_panc", "EAURC_kidn", "AURC_livr", "ECE_1", "ECE_2", "ECE_3", "ACE_1", "ACE_2", "ACE_3", "CRPS_panc", "CRPS_kidn", "CRPS_livr", "NCC"])

#Computing the metrics

for f in l_patients : 
	current_line = pd.DataFrame(apply_metrics(f))
	df = pd.concat([df,current_line], ignore_index=True)

#Export data
#df.to_csv("metrics.csv", index=False)
print(df)