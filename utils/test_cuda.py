import torch
import subprocess as sp
import os
import time

has_cuda = torch.cuda.is_available()
print("Cuda available:", has_cuda)

def pick_gpu(wait=True):

    command = "nvidia-smi --query-gpu=memory.free --format=csv"
    memory_free_info = sp.check_output(command.split()).decode('ascii').split('\n')[:-1][1:]
    memory_free_values = [int(x.split()[0]) for i, x in enumerate(memory_free_info)]
    while True:
        for idx in range(len(memory_free_values)):
            print(memory_free_values[idx])
            if memory_free_values[idx] >= 45000:
                print(f"using GPU {idx}")
                os.environ["CUDA_VISIBLE_DEVICES"] = str(idx)
                return
        
        print(f"GPU not available")
        if wait:
            print("Trying again in 10 minutes...")
            time.sleep(600)
        else:
            return

# pick_gpu()

