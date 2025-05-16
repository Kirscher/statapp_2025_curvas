import os
from Metrics_func.new_metrics import apply_metrics, getting_gt
import boto3
import pandas as pd

# inputs
aws_access_key_id = "MPOO6QLTXGQVW61QK4ZE"#input(str("aws_access_key_id : "))
aws_secret_access_key = "LeZYGLEp99y+IkkKBRkRaNVdkc2YHK+lekNx+L+f"#input(str("aws_access_key_key : "))
aws_session_token = "eyJhbGciOiJIUzUxMiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3NLZXkiOiJNUE9PNlFMVFhHUVZXNjFRSzRaRSIsImFsbG93ZWQtb3JpZ2lucyI6WyIqIl0sImF1ZCI6WyJtaW5pby1kYXRhbm9kZSIsIm9ueXhpYSIsImFjY291bnQiXSwiYXV0aF90aW1lIjoxNzQ2OTEwOTAxLCJhenAiOiJvbnl4aWEiLCJlbWFpbCI6Imx1Y2FzLmN1bXVuZWxAZW5zYWUuZnIiLCJlbWFpbF92ZXJpZmllZCI6dHJ1ZSwiZXhwIjoxNzQ3NTE1ODk5LCJmYW1pbHlfbmFtZSI6IkN1bXVuZWwiLCJnaXZlbl9uYW1lIjoiTHVjYXMiLCJncm91cHMiOlsiVVNFUl9PTllYSUEiLCJzdGF0YXBwLXNlZ21lZGljIl0sImlhdCI6MTc0NjkxMTA5OSwiaXNzIjoiaHR0cHM6Ly9hdXRoLmxhYi5zc3BjbG91ZC5mci9hdXRoL3JlYWxtcy9zc3BjbG91ZCIsImp0aSI6Im9ucnRydDo2NWE4Zjc1ZS03NjZlLTRhZmEtYTJjZC03ZjQwMzY0MjI4OWYiLCJuYW1lIjoiTHVjYXMgQ3VtdW5lbCIsInBvbGljeSI6InN0c29ubHkiLCJwcmVmZXJyZWRfdXNlcm5hbWUiOiJsYWIiLCJyZWFsbV9hY2Nlc3MiOnsicm9sZXMiOlsib2ZmbGluZV9hY2Nlc3MiLCJ1bWFfYXV0aG9yaXphdGlvbiIsImRlZmF1bHQtcm9sZXMtc3NwY2xvdWQiXX0sInJlc291cmNlX2FjY2VzcyI6eyJhY2NvdW50Ijp7InJvbGVzIjpbIm1hbmFnZS1hY2NvdW50IiwibWFuYWdlLWFjY291bnQtbGlua3MiLCJ2aWV3LXByb2ZpbGUiXX19LCJyb2xlcyI6WyJvZmZsaW5lX2FjY2VzcyIsInVtYV9hdXRob3JpemF0aW9uIiwiZGVmYXVsdC1yb2xlcy1zc3BjbG91ZCJdLCJzY29wZSI6Im9wZW5pZCBwcm9maWxlIGdyb3VwcyBlbWFpbCIsInNpZCI6IjA3ODg3YzIwLTFjMmUtNDliMC1iMTg5LWE4OGU0OGYxM2NjNCIsInN1YiI6ImUyZDc4NjRjLTcwMzItNDI0ZC04OTA2LWU0ZjhiNDFjYzAwMyIsInR5cCI6IkJlYXJlciJ9.ga6x727DgquSCMmy-I609uoqDHAw0zGcPBN2ayD8PmE_Gp9sSerqAgJ5Yfqxl1XmGq6WWzqPeDvnxBYIBpgWyw"#input(str("aws_session_token : "))
patient = "UKCHLL007"#input(str("Patient ID (name of the folder with all the models) :"))
pred_patient_path = "projet-statapp-segmedic/output/UKCHLL007"#input(str("Path to folder with predictions in s3 (only bucket/.../folder) :"))
gt_patient_path = "projet-statapp-segmedic/data/UKCHLL007"#input(str("Path to folder with GT in s3 (only bucket/.../folder) :"))
#output_path = input(str("Path to storage for metrics output (only bucket/.../folder) :"))

s3 = boto3.client(
    "s3",
    endpoint_url='https://minio.lab.sspcloud.fr',
    aws_access_key_id=aws_access_key_id,
    aws_secret_access_key=aws_secret_access_key,
    aws_session_token=aws_session_token
)


pred_command = f"aws s3 cp s3://{pred_patient_path} ./{patient} --endpoint-url https://minio.lab.sspcloud.fr --recursive"
gt_command = f"aws s3 cp s3://{gt_patient_path} ./{patient}/{patient}_GT --endpoint-url https://minio.lab.sspcloud.fr --recursive"
output_command = f"aws s3 cp ./metrics.csv s3://metrics_results/{patient}/metrics.csv --endpoint-url https://minio.lab.sspcloud.fr"

# s3.upload_file("metrics.csv", "projet-statapp-segmedic", "metrics_results/patient/metrics.csv")

os.system(pred_command)
os.system(gt_command)


# Locating data
data_path = f"./{patient}"
l_models_path = os.listdir(data_path)
l_models_path = [item for item in l_models_path if "GT" not in item]
l_models = []

for model_dir in l_models_path:
    model_dict = {"pred": None, "prob": None, "name": re.findall(r"\/([^\/]+)$",model_dir)}
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
#TODO, prio ensemble
for f in l_models: 
    current_line = pd.DataFrame([apply_metrics(f, getting_gt(gt_path))])
    df = pd.concat([df, current_line], ignore_index=True)

# Export data
df.to_csv("metrics.csv", index=False)
print(df)
#os.system(output_command)
