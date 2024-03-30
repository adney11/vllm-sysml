#!/bin/bash
function enable_mps_if_needed()
{
    mode=$1
    if [[ ${mode} == mps-* ]]; then
        echo "Enabling MPS"
        nvidia-smi -i 0 -c DEFAULT  
        nvidia-cuda-mps-control -d

        if [[ $(ps -eaf | grep nvidia-cuda-mps-control | grep -v grep | wc -l) == 0 ]]; then
            echo "Unable to enable MPS"
            exit 1
        fi
    fi
}

function disable_mps_if_needed()
{
    mode=$1
    if [[ ${mode} == mps-* ]]; then
        echo "Disabling MPS"
        echo quit | nvidia-cuda-mps-control
        nvidia-smi -i 0 -c DEFAULT

        if [[ $(ps -eaf | grep nvidia-cuda-mps-control | grep -v grep | wc -l) -ne 0 ]]; then
            echo "Unable to disable MPS"
            exit 1
        fi
    fi
}


enable_mps_if_needed mps-*

echo "Running script"
python llm_engine_mps_test.py --gpu-memory-utilization 0.45 > instance1.output &
# echo "sleeping before starting next instance"
# sleep 4
python llm_engine_mps_test.py --gpu-memory-utilization 0.45 > instance2.output

disable_mps_if_needed mps-*