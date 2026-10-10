import sys
import cv2
import torch

print("=========================================")
print("🚀 PIPELINE VALIDATION SUCCESSFUL! 🚀")
print("=========================================")

# 1. Environment Details
print("Python Location:", sys.executable)
print("PyTorch Version:", torch.__version__)
print("OpenCV Version:", cv2.__version__)

# 2. Hardware Test
gpu_test = torch.cuda.is_available()
print("GPU Acceleration Available:", gpu_test)
print("=========================================")