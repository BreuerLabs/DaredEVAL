import torch
import subprocess as sp
import os

has_cuda = torch.cuda.is_available()
print("Cuda available:", has_cuda)

def pick_gpu(wait_one_gpu=False, gpu_idx=0):

    command = "nvidia-smi --query-gpu=memory.free --format=csv"
    memory_free_info = sp.check_output(command.split()).decode('ascii').split('\n')[:-1][1:]
    memory_free_values = [int(x.split()[0]) for i, x in enumerate(memory_free_info)]
    for idx in range(len(memory_free_values)):
        print(memory_free_values[idx])
        if memory_free_values[idx] == 48670:
            print(f"using GPU {idx}")
            os.environ["CUDA_VISIBLE_DEVICES"] = str(idx)
            return
    print(f"GPUs not available")

pick_gpu()

