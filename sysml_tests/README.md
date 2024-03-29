# Profiling
nsys --version: NVIDIA Nsight Systems version 2024.1.1.59-241133802077v0  
nvcc --version: release 12.3, V12.3.107

To start:
1. `cd SysML-splitwiser/repos/vllm-sysml`  
2. Look over *sysml_tests/llm_engine_example_profile.py*:  
    - The engine is created w/ CLI args. Primarily you'll need to pass in *--model [model]* ex. *--model facebook/opt-125m* or *--model meta-llama/llama-2-7b-hf*  
    - Globals at the top of file can be changed to control in/out tokens.  
3. You may need to set this: https://docs.nvidia.com/nsight-systems/InstallationGuide/index.html#requirements-for-x86-64-power-and-arm-sbsa-targets-on-linux  

## ncu
Preferred profiling method. *Note: There is significant overhead, particularly w/ larger output tokens. Start w/ less output tokens (ex. 5), then add as needed.*
1. Configure script with desired # input/output tokens and the *.csv that corresponds to the model used and contains prompts w/ correct # of tokens.  
2. Run script w/ ncu profiling:  
    - Ex. `psudo HF_HOME=/sysml/.cache TMPDIR=/sysml/tmp CUDA_VISIBLE_DEVICES=0 ncu --replay-mode application --nvtx --nvtx-include "prompt/" --nvtx-include "token/" --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed,dram__throughput.avg.pct_of_peak_sustained_elapsed,dram__bytes_read.sum.pct_of_peak_sustained_elapsed,dram__bytes_write.sum.pct_of_peak_sustained_elapsed -o tok5_a10_llama7b_single_req_ncu_sm_mem python sysml_tests/llm_engine_example_profile.py --model meta-llama/llama-2-7b-hf`  
    - `--target-processes all`: Target root process and child processes (for multi-ranks)
3. Process the generated .ncu-rep w/ the `SysML-splitwiser/utils/parse_ncu_rep.ppy` script  


# Running MPS
We will be using the `llm_engine_mps_test.py` script in this directory as the MPS instance code.
Simply run `run_unmodified_llm_engine_mps.sh` which will enable MPS, start the first instance, wait 2 seconds and start the next instance, and finally disable MPS.
The outputs of the instances can be found in `instance1.output` and `instance2.output`