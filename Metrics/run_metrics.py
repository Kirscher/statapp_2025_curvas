# pip install scikit-learn torch scipy monai torchmetrics numba boto3 SimpleITK awscli

import os
from Metrics_func.new_metrics import apply_metrics, getting_gt
import boto3
import pandas as pd
import subprocess

# inputs
# inputs
aws_access_key_id = input("aws_access_key_id : ")
aws_secret_access_key = input("aws_secret_access_key : ")
aws_session_token = input("aws_session_token : ")
patient = input("Patient ID (name of the folder with all the models) :")


skip = input("skip download ? (yes/no)")
if skip != "yes":
    # Setup boto3 client (optional if you only use CLI commands)
    s3 = boto3.client(
        "s3",
        endpoint_url='https://minio.lab.sspcloud.fr',
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        aws_session_token=aws_session_token
    )
    
    # Environment variables for AWS CLI
    env = os.environ.copy()
    env["AWS_ACCESS_KEY_ID"] = aws_access_key_id
    env["AWS_SECRET_ACCESS_KEY"] = aws_secret_access_key
    env["AWS_SESSION_TOKEN"] = aws_session_token
    
    # Commands
    pred_command = [
        "aws", "s3", "cp", 
        f"s3://projet-statapp-segmedic/output/{patient}", 
        f"./{patient}", 
        "--endpoint-url", "https://minio.lab.sspcloud.fr", 
        "--recursive"
    ]
    
    gt_command = [
        "aws", "s3", "cp", 
        f"s3://projet-statapp-segmedic/data/{patient}", 
        f"./{patient}/{patient}_GT", 
        "--endpoint-url", "https://minio.lab.sspcloud.fr", 
        "--recursive"
    ]
    
    output_command = [
        "aws", "s3", "cp", 
        "./metrics.csv", 
        f"s3://projet-statapp-segmedic/metrics_results/{patient}/metrics.csv", 
        "--endpoint-url", "https://minio.lab.sspcloud.fr"
    ]
    
    # Execute CLI commands with injected credentials
    subprocess.run(pred_command, env=env)
    subprocess.run(gt_command, env=env)


# Locating data
data_path = f"./{patient}"
l_models_path = os.listdir(data_path)
l_models_path = [item for item in l_models_path if "GT" not in item]
l_models_path = sorted(l_models_path, key=lambda x: "ensembl" not in x)
l_models = []

for model_dir in l_models_path:
    model_dict = {"pred": None, "prob": None, "name": os.path.basename(model_dir)}
    model_path = os.path.join(data_path, model_dir)
    for item in os.listdir(model_path):
        item_path = os.path.join(model_path, item)
        if item.endswith(".nii.gz"):
            model_dict["pred"] = item_path
        elif item.endswith(".npz"):
            model_dict["prob"] = item_path
    l_models.append(model_dict)
gt_path = os.path.join(data_path, os.path.basename(str(data_path+"_GT")))

# Prepare output
df = pd.DataFrame(columns=[
        "Patient_ID",
        "CT",
        "DICE_panc",
        "DICE_kidn",
        "DICE_livr",
        "CONF_panc",
        "CONF_kidn",
        "CONF_livr",
        "Entropy_GT",
        "Entropy_Pred",
        "Hausdorff_1",
        "Hausdorff_2",
        "Hausdorff_3",
        "ECE_1",
        "ECE_2",
        "ECE_3",
        "ACE_1",
        "ACE_2",
        "ACE_3",
        "AUROC_panc",
        "AUROC_kidn",
        "AUROC_livr",
        "AURC_panc",
        "AURC_kidn",
        "AURC_livr",
        "EAURC_panc",
        "EAURC_kidn",
        "EAURC_livr",
        "CRPS_panc",
        "CRPS_kidn",
        "CRPS_livr",
        "NCC_GT1-2",
        "NCC_GT1-3",
        "NCC_GT2-3",
        "NCC_mean"
    ])

# Computing the metrics

ct_img, annot = getting_gt(gt_path)
for f in l_models: 
    current_line = pd.DataFrame([apply_metrics(f, ct_img, annot)])
    df = pd.concat([df, current_line], ignore_index=True)
df['Patient_ID'] = patient

# Export data
df.to_csv("metrics.csv", index=False)
print(df)

# Output in s3
subprocess.run(output_command, env=env)
