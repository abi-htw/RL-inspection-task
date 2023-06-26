import torch
import numpy as np

validation = []
loaded_tensor_list = []

loaded_numpy_list = np.load('tensors.npy', allow_pickle=True)

# Convert the NumPy arrays back to PyTorch tensors
for item in loaded_numpy_list:
    loaded_tensor_list.append([torch.from_numpy(numpy_array) for numpy_array in item])

# Verify the loaded list
print(loaded_tensor_list[0])