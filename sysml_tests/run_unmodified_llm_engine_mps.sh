#!/bin/bash
function enable_mps_if_needed()
{
    mode=$1
    if [[ ${mode} == mps-* ]]; then
        echo "Enabling MPS"
        nvidia-smi -i 0 -c EXCLUSIVE_PROCESS
        nvidia-cuda-mps-control -d

        if [[ $(ps -eaf | grep nvidia-cuda-mps-control | grep -v grep | wc -l) -ne 1 ]]; then
            echo "Unable to enable MPS"
            exit 1
        fi
    fi
}

function disable_mps_if_needed()
{
    mode=$1
    if [[ ${mode} == mps-* ]]; then
        echo quit | nvidia-cuda-mps-control
        nvidia-smi -i 0 -c DEFAULT

        if [[ $(ps -eaf | grep nvidia-cuda-mps-control | grep -v grep | wc -l) -ne 0 ]]; then
            echo "Unable to disable MPS"
            exit 1
        fi
    fi
}

echo "enabling MPS.."
enable_mps_if_needed

python llm_engine_mps_test.py --gpu-memory-utilization 0.45 --enforce-eager > instance1.output &
echo "sleeping 2 seconds before starting next instance"
sleep 2
python llm_engine_mps_test.py --gpu-memory-utilization 0.45 --enforce-eager > instance2.output

echo "disabling MPS..."
disable_mps_if_needed