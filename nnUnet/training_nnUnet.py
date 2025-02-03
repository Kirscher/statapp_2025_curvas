# Training nnUnet

import torch
#pip install nnunetv2


x = torch.tensor([1.0, 2.0, 3.0])
print(x)
x_gpu = torch.randn(3, 3).cuda()
print(x_gpu)

import nnunetv2
#nnU-Net needs environment variables, write the following inside the terminal
# export nnUNet_raw="/home/onyxia/work/statapp_2025_curvas/nnUnet/dataset/nnUNet_raw"
# export nnUNet_preprocessed="/home/onyxia/work/statapp_2025_curvas/nnUnet/dataset/nnUNet_preprocessed"
# export nnUNet_results="/home/onyxia/work/statapp_2025_curvas/nnUnet/dataset/nnUNet_results"