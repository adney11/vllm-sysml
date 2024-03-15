## nsys
Not preferred. May randomly fail with changing system/tests.
User guide/CLI Info: https://docs.nvidia.com/nsight-systems/UserGuide/index.html#cli-launch-command-switch-options 
1. `cd SysML-splitwiser/repos/vllm-sysml`  
2. I had to run nsys profile as root user to access GPU resources. Additionally,  I had to add python back to the PATH. If this is needed for you, you can add this in ~/.bashrc: `psudo() { sudo env PATH="$PATH" "$@"; }`  
3. Start nsys and redirect output to log file (**make sure to do (1) `nsys start` before any (2) `nsys launch`**):  
    1. With *psudo* (otherwise replace with *sudo*), nsys start:  
    `psudo nsys start -f true -c cudaProfilerApi --gpu-metrics-device=0 -o [gpu]_[model]_[n]itokens_[n]otokens`  
        - Ex. (On Oracle)  
        `psudo nsys start -f true -c cudaProfilerApi --gpu-metrics-device=0 -o a10_opt125m_1024itokens_1024otokens`  
    2. With *psudo* (otherwise replace with *sudo*), nsys launch (**this redirects all output to .log. If want to see stdout, remove the redirection part**):  
    `psudo nsys launch -w true python sysml_tests/llm_engine_example_profile.py --model [model] > [gpu]_[model]_[prompt/token]_[n]itokens_[n]otokens.log 2>&1`  
        - Ex. (On Oracle)  
        `psudo nsys launch -w true python sysml_tests/llm_engine_example_profile.py --model facebook/opt-125m > a10_opt125m_1024itokens_1024otokens.log 2>&1`  
4. Share the generated *.nsys-rep and *.log files in the Google Drive shared folder.  
5. You can view results by opening the *.nsys-rep file in NVIDIA Nsight Systems GUI: https://developer.nvidia.com/nsight-systems/get-started. You can also view stats: `nsys stats --format csv [name].nsys-rep --output .`

(Alternative #3, profile phases separately) Start nsys and redirect output to log file:  
1. With *psudo* (otherwise replace with *sudo*), nsys start:  
`psudo nsys start -c nvtx --gpu-metrics-device=0 -o [gpu]_[model]_[prompt/token]_[n]itokens_[n]otokens`  
    - Ex. (On Oracle)
    `psudo nsys start -c nvtx --gpu-metrics-device=0 -o a10_opt125m_token_1024itokens_1024otokens`  
2. With *psudo* (otherwise replace with *sudo*), nsys launch:  
`psudo nsys launch -w true -p [prompt/token] -e NSYS_NVTX_PROFILER_REGISTER_ONLY=0 python sysml_tests/llm_engine_example_profile.py --model facebook/opt-125m > [gpu]_[model]_[prompt/token]_[n]itokens_[n]otokens.log 2>&1`  
    - Ex. (On Oracle)  
    `psudo nsys launch -w true -p prompt -e NSYS_NVTX_PROFILER_REGISTER_ONLY=0 python sysml_tests/llm_engine_example_profile.py --model facebook/opt-125m > a10_opt125m_token_1024itokens_1024otokens.log 2>&1`  
Re-run steps 1&2 to capture both prompt/token phase in separate traces.

## (Optional) nvidia-smi 
Possibly not preferred, not too informative and minimum 1 ms sampling period.
1. `cd SysML-splitwiser/repos/vllm-sysml`  
2. In *sysml_tests/llm_engine_example_profile.py* you'll need to set `ENABLE_PROFILING_PAUSES = True`
3. `python sysml_tests/llm_engine_example_profile.py --model [model]`  
4. You will see **===> [Start/End] of [PROMPT/TOKEN] phase: [Begin/Stop] profiling...** 
    - Do profiling on a second shell  
    - You'll be switching between shells to (1) begin/end profiling and (2) continue script execution (enter/any key)  
    - Start: Begin nvidia-smi cmd  
        - `nvidia-smi -i 0 --query-gpu=timestamp,driver_version,name,index,pstate,power.draw.instant,clocks.current.sm,clocks.current.memory,memory.total,memory.free,memory.used,utilization.gpu,utilization.memory --format=csv -lms 1 -f [n]tokens_[prompt/token]_phase_nvidia-smi.csv`  
        - nvidia-smi: https://developer.download.nvidia.com/compute/DCGM/docs/nvidia-smi-367.38.pdf  
    - End: Stop nvidia-smi cmd (CTRL+C)  

## (Optional) nvidia-smi pmon
Min 1 second sampling period.
Same steps as above nvidia-smi approach, just replace the nvidia-smi cmd with:
`nvidia-smi dmon -i 0 -s pucvmet -o T -f [n]tokens_[prompt/token]_phase_nvidia-smi.csv`
