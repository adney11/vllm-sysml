import torch
import threading
import argparse
from vllm import EngineArgs
from llm_engine_example_single import run
import copy

var1 = 10
var2 = 20

def worker(lock, rank):
    var1_tensor = torch.tensor(var1)  # Assuming rank is known within the thread
    var2_tensor = torch.tensor(var2)

    # Thread-safe operations using locks or shared memory access

    if rank == 1:
        var1_tensor = var2_tensor  # Update var1_tensor based on communication

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Demo on using the LLMEngine class directly')
    parser = EngineArgs.add_cli_args(parser)
    args = parser.parse_args()
    args_phase2 = copy.deepcopy(args)

    # creating a lock 
    lock = threading.Lock() 
    
    # Create threads with appropriate rank information
    # thread1 = threading.Thread(target=run, args=(args,))
    thread1 = threading.Thread(target=worker, args=(lock, 0,))
    
    args_phase2.is_token_phase = True
    # thread2 = threading.Thread(target=run, args=(args_phase2,))
    thread2 = threading.Thread(target=worker, args=(lock, 1,))

    # Start threads
    thread1.start()
    thread2.start()

    # Wait for threads to finish
    thread1.join()
    thread2.join()

# Process received data in the main thread based on rank information

# """run.py:"""
# #!/usr/bin/env python
# import os
# import torch
# import torch.distributed as dist
# import torch.multiprocessing as mp
# import sys

# """Blocking point-to-point communication."""

# def run(rank, size):
#     tensor = torch.zeros(1).cuda()
#     if rank == 0:
#         tensor += 1
#         # Send the tensor to process 1
#         dist.send(tensor=tensor, dst=1)
#     else:
#         # Receive tensor from process 0
#         dist.recv(tensor=tensor, src=0)
#     print('Rank ', rank, ' has data ', tensor[0])

# def init_process(rank, size, fn, backend='gloo'):
#     """ Initialize the distributed environment. """
#     os.environ['MASTER_ADDR'] = '127.0.0.1'
#     os.environ['MASTER_PORT'] = '29500'
#     dist.init_process_group(backend, rank=rank, world_size=size)
#     fn(rank, size)


# if __name__ == "__main__":
#     size = 2
#     processes = []
    
#     rank = int(sys.argv[1])
#     init_process(rank, size, run)

#     # mp.set_start_method("spawn")
#     # for rank in range(size):
#     #     p = mp.Process(target=init_process, args=(rank, size, run))
#     #     p.start()
#     #     processes.append(p)

#     # for p in processes:
#     #     p.join()