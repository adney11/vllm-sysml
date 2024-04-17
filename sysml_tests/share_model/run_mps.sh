#!/bin/bash
function enable_mps_if_needed()
{
    echo "Enabling MPS"
    # export CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=25
    # nvidia-smi -i 3 -c DEFAULT
    nvidia-smi -i 3 -c EXCLUSIVE_PROCESS
    nvidia-cuda-mps-control -d

    if [[ $(ps -eaf | grep nvidia-cuda-mps-control | grep -v grep | wc -l) -ne 1 ]]; then
        echo "Unable to enable MPS"
        exit 1
    fi
}

function disable_mps_if_needed()
{
    echo quit | nvidia-cuda-mps-control
    nvidia-smi -i 3 -c DEFAULT

    if [[ $(ps -eaf | grep nvidia-cuda-mps-control | grep -v grep | wc -l) -ne 0 ]]; then
        echo "Unable to disable MPS"
        exit 1
    fi
}

# echo "enabling MPS.."
enable_mps_if_needed

HF_HOME=/sysml/.cache TMPDIR=/sysml/tmp CUDA_VISIBLE_DEVICES=3 nsys profile --capture-range cudaProfilerApi --gpu-metrics-device=3 -o check_llm_mps -f true python share_model.py
# HF_HOME=/sysml/.cache TMPDIR=/sysml/tmp CUDA_VISIBLE_DEVICES=3 python share_model.py

# # echo "disabling MPS..."
disable_mps_if_needed