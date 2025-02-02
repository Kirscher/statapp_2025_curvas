# Training nnUnet

import torch
#pip install nnunetv2


x = torch.tensor([1.0, 2.0, 3.0])
print(x)
x_gpu = torch.randn(3, 3).cuda()
print(x_gpu)

#nnU-Net needs environment variables
import nnunetv2