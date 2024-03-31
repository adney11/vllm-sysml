import argparse
from typing import List, Tuple

from vllm import EngineArgs, LLMEngine, SamplingParams, RequestOutput
import torch

import csv
import time
import os
import pathlib
tests_path = pathlib.Path(__file__).parent.resolve()

ENABLE_PROFILING_PAUSES = False
PROFILE_SECTION = "inf" # "prompt", "token", "step", "inf", "not_nsys"
NUM_OUTPUT_TOKENS = 1024 # sets max_tokens, so it is upper limit not a guaranteed output length
NUM_INPUT_TOKENS = 1024 # selects randomly generated prompt with n tokens from random_prompts.csv where n = [128, 256, 512, 1024, 2048, 4096, 8192]

def create_test_prompts() -> List[Tuple[str, SamplingParams]]:
    """Create a list of test prompts with their sampling parameters."""
    return [
        ("### findings: heart size within normal limits. no focal alveolar consolidation, no definite pleural effusion seen. no typical findings of pulmonary edema. mediastinal calcification and dense right upper lung nodule suggest a previous granulomatous process. ### impressions:",
         SamplingParams(n=1, temperature=0.0, logprobs=1, prompt_logprobs=1, max_tokens=1024)),
    ]


def process_requests(engine: LLMEngine,
                     test_prompts: List[Tuple[str, SamplingParams]]):
    """Continuously process a list of prompts and handle the outputs."""
    request_id = 0
    step = 0

    while test_prompts or engine.has_unfinished_requests():
        if test_prompts:
            prompt, sampling_params = test_prompts.pop(0)
            engine.add_request(str(request_id), prompt, sampling_params)
            request_id += 1

        if step == 0: # step 0 = prompt processing step:
            print(f"===> Start of PROMPT phase: Begin profiling...")
            if ENABLE_PROFILING_PAUSES:
                input()
            # torch.cuda.nvtx.mark("START PROMPT")
            if PROFILE_SECTION == "inf" or PROFILE_SECTION == "prompt":
                torch.cuda.cudart().cudaProfilerStart()
            torch.cuda.nvtx.range_push("prompt")
            engine.stat_logger.start_log_point(time.monotonic())
        elif step == 1:
            print(f"===> Start of TOKEN phase: Begin profiling...")
            if ENABLE_PROFILING_PAUSES:
                input()
            # torch.cuda.nvtx.mark("START TOKEN")
            if PROFILE_SECTION == "token":
                torch.cuda.cudart().cudaProfilerStart()
            torch.cuda.nvtx.range_push("token")
            engine.stat_logger.start_log_point(time.monotonic())

        request_outputs: List[RequestOutput] = engine.step()

        if step == 0: # step 0 = token processing step:
            print(f"===> End of PROMPT phase: Stop profiling...")
            # torch.cuda.nvtx.mark("END PROMPT")
            engine.stat_logger.end_log_point(time.monotonic())
            torch.cuda.nvtx.range_pop()
            if PROFILE_SECTION == "prompt":
                torch.cuda.cudart().cudaProfilerStop()
            if ENABLE_PROFILING_PAUSES:
                input()
        elif not engine.has_unfinished_requests():
            print(f"===> End of TOKEN phase: Stop profiling...")
            # torch.cuda.nvtx.mark("END TOKEN")
            engine.stat_logger.end_log_point(time.monotonic())
            torch.cuda.nvtx.range_pop()
            if PROFILE_SECTION == "inf" or PROFILE_SECTION == "token":
                torch.cuda.cudart().cudaProfilerStop()
            if ENABLE_PROFILING_PAUSES:
                input()

        step += 1

        if request_outputs == None:
            continue
        
        for request_output in request_outputs:
            # if request_output.finished:
            #     print(request_output)
            if request_output.finished:
                print(f"Request ID: {request_output.request_id} \
                      # Input Tokens: {len(request_output.prompt_token_ids)} \
                      # Output Tokens: {len(request_output.outputs[0].token_ids)} \
                      Finish Reason: {request_output.outputs[0].finish_reason} \n \
                      Output Text: {request_output.outputs[0].text!r}")
            
    print(f"\n===! Num steps executed: {step} !===\n")


def initialize_engine(args: argparse.Namespace) -> LLMEngine:
    """Initialize the LLMEngine from the command line arguments."""
    engine_args = EngineArgs.from_cli_args(args)
    return LLMEngine.from_engine_args(engine_args)


def run(args: argparse.Namespace):
    """Main function that sets up and runs the prompt processing."""
    args.model = "facebook/opt-125m"
    engine = initialize_engine(args)
    test_prompts = create_test_prompts()
    process_requests(engine, test_prompts)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Demo on using the LLMEngine class directly')
    parser = EngineArgs.add_cli_args(parser)
    args = parser.parse_args()
    run(args)
