import torch
import torch.multiprocessing as mp
# import threading
import time
import tempfile
import argparse

from vllm import EngineArgs, LLMEngine, SamplingParams, RequestOutput
from vllm.config import (ModelConfig, ParallelConfig, SchedulerConfig,
                         DeviceConfig, LoRAConfig)
from vllm.model_executor.model_loader import get_model
from vllm.model_executor.parallel_utils.parallel_state import (
    destroy_model_parallel, initialize_model_parallel)

import vllm_worker as vwork

num_workers = 2
shared_model = None

def init(model="meta-llama/llama-2-7b-hf"):
    global shared_model

    if not torch.distributed.is_initialized():
        temp_file = tempfile.mkstemp()[1]
        torch.distributed.init_process_group(
            backend="nccl",
            world_size=1,
            rank=0,
            init_method=f"file://{temp_file}",
        )
    initialize_model_parallel(1, 1)
    tokenizer = model
    model_config=ModelConfig(
        model,
        tokenizer,
        tokenizer_mode="auto",
        trust_remote_code=False,
        download_dir=None,
        load_format="auto",
        seed=0,
        dtype="float16",
        revision=None,
    )
    device_config=DeviceConfig("cuda")

    shared_model = get_model(model_config, device_config, lora_config=None)

def cleanup():
    destroy_model_parallel()
    torch.cuda.empty_cache()

### Threading
# sem = threading.Semaphore(0)
def worker(rank, model, args):
    print(f"Creating worker for rank {rank}")
    kwargs = {
        "rank": rank,
        "shared_model": model,
    }
    vwork.worker_main(args, **kwargs)
    # if rank == 0:
    #     print("Setting model supports_lora = False")
    #     shared_model.supports_lora = False
    #     sem.release()
    # else:
    #     sem.acquire()
    #     print(f"Checking supports_lora set to false: {shared_model.supports_lora}")

if __name__ == '__main__':
    init()

    parser = argparse.ArgumentParser(
        description='Demo on using the LLMEngine class directly')
    parser = EngineArgs.add_cli_args(parser)
    args = parser.parse_args()

    ### torch.mp 
    mp.set_start_method("spawn", force=True)
    shared_model.share_memory()
    workers = []

    for rank in range(num_workers):
        w = mp.Process(target=worker, args=(rank, shared_model, args), name=f'Inf_Worker_{rank}')
        workers.append(w)

    start = time.time()
    for w in workers:
        w.start()

    for w in workers:
        w.join()
    stop = time.time()

    ### Threading
    # t1 = threading.Thread(target=worker, args=(0, args))
    # t2 = threading.Thread(target=worker, args=(1, args))

    # # Start threads
    # start = time.time()
    # t1.start()
    # # t2.start()

    # # Wait for threads to finish
    # t1.join()
    # # t2.join()
    # stop = time.time()

    print(f"Completed in {stop-start} s")

    cleanup()