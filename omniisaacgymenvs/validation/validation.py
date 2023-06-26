# from omni.isaac.core.utils.torch import torch_rand_float
import torch
import numpy as np

@torch.jit.script
def torch_rand_float(lower, upper, shape, device):
    # type: (float, float, Tuple[int, int], str) -> Tensor
    return (upper - lower) * torch.rand(*shape, device=device) + lower




def newpos():
    return torch_rand_float(-0.7, 0.7 , (1, 3), device="cuda")




validation = [[newpos() for _ in range(6)] for x in range(200)]

# with open(r'/RLrepo/OmniIsaacGymEnvs-UR10Reacher/omniisaacgymenvs/validation/val.txt', 'w') as fp:
#     for item in validation:
#         # write each item on a new line
#         fp.write("%s\n" % item)
#     print('Done')

numpy_list = []

for element in validation:
    numpy_list.append([tensor.cpu().numpy() for tensor in element])

# Save the NumPy array to a file
np.save('tensors.npy', numpy_list)