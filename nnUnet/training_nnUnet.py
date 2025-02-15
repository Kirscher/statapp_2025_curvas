# Training nnUnet

# to download later:
#(OPTIONAL) Install hiddenlayer. hiddenlayer enables nnU-net to generate plots of the network 
# topologies it generates (see Model training). To install hiddenlayer, run the following command:

import torch
#pip3 install nnunetv2
import s3fs
import os


# Connexion à MinIO S3 Onyxia
s3 = s3fs.S3FileSystem(
    client_kwargs={'endpoint_url': 'https://'+'minio.lab.sspcloud.fr'},
    key=os.getenv("AWS_ACCESS_KEY_ID"),
    secret=os.getenv("AWS_SECRET_ACCESS_KEY"),
    token=os.getenv("AWS_SESSION_TOKEN")
)

nnUNet_raw = "s3://leoacpr/diffusion/nnunet_dataset/nnUNet_raw"
nnUNet_preprocessed = "s3://leoacpr/diffusion/nnunet_dataset/nnUNet_preprocessed"
nnUNet_results = "s3://leoacpr/diffusion/nnunet_dataset/nnUNet_results"


'''x = torch.tensor([1.0, 2.0, 3.0])
print(x)
x_gpu = torch.randn(3, 3).cuda()
print(x_gpu)'''

#import nnunetv2
#nnU-Net needs environment variables, write the following inside the terminal
# export nnUNet_raw="/home/onyxia/work/statapp_2025_curvas/nnUnet/dataset/nnUNet_raw"
# export nnUNet_preprocessed="/home/onyxia/work/statapp_2025_curvas/nnUnet/dataset/nnUNet_preprocessed"
# export nnUNet_results="/home/onyxia/work/statapp_2025_curvas/nnUnet/dataset/nnUNet_results"

#or 
# s3fs leoacpr/diffusion /mnt/onyxia_s3 -o iam_role=auto
# export nnUNet_raw="/mnt/onyxia_s3/nnunet_dataset/nnUNet_raw"
# export nnUNet_preprocessed="/mnt/onyxia_s3/nnunet_dataset/nnUNet_preprocessed"
# export nnUNet_results="/mnt/onyxia_s3/nnunet_dataset/nnUNet_results"
