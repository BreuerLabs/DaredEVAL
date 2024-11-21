import torch

has_cuda = torch.cuda.is_available()
print("Cuda available:", has_cuda)

def pick_gpu(wait_one_gpu=False, gpu_idx=0):
    command = "nvidia-smi --query-gpu=memory.free --format=csv"
            memory_free_info = sp.check_output(command.split()).decode('ascii').split('\n')[:-1][1:]
            memory_free_values = [int(x.split()[0]) for i, x in enumerate(memory_free_info)]
            if memory_free_values[gpu_idx] == 48676:
                print(f"using GPU {gpu_idx}")
                os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_idx)
                return

        while True:
            command = "nvidia-smi --query-gpu=memory.free --format=csv"
            memory_free_info = sp.check_output(command.split()).decode('ascii').split('\n')[:-1][1:]
            memory_free_values = [int(x.split()[0]) for i, x in enumerate(memory_free_info)]
            for idx in range(len(memory_free_values)):
                if memory_free_values[idx] == 48676:
                    print(f"using GPU {idx}")
                    os.environ["CUDA_VISIBLE_DEVICES"] = str(idx)
                    return
            print(f"GPUs not available, sleeping for 60 minutes")
            time.sleep(1800)
            print("30 minutes left")
            time.sleep(1800)