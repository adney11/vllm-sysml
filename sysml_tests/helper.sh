#!/bin/bash

VENV=~/venv/

function assert_mig_status()
{
    mode=$1
    if [[ ${mode} == "mig" ]]; then
        if [[ $(nvidia-smi -i ${DEVICE_ID} --query-gpu=mig.mode.current --format=csv | grep "Enabled" | wc -l) -eq 0 ]]; then
            echo "MIG mode not enabled"
            exit 1
        fi
    else
        if [[ $(nvidia-smi -i ${DEVICE_ID} --query-gpu=mig.mode.current --format=csv | grep "Disabled" | wc -l) -eq 0 ]]; then
            echo "MIG mode is not disabled"
            exit 1
        fi
    fi
}

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

function setup_mig_if_needed()
{
    mode=$1
    num_procs=$2
    if [[ ${mode} != "mig" ]]; then
        return
    fi

    if [[ $# -eq 3 && $3 == "chunks" ]]; then
        gi=("" "19" "19,19" "19,19,19" "19,19,19,19" "19,19,19,19,19" "19,19,19,19,19,19" "19,19,19,19,19,19,19")
    else
        gi=("" "0" "9,5" "9,14,14" "14,14,14,19" "14,14,19,19,19" "14,19,19,19,19,19" "19,19,19,19,19,19,19")
    fi


    # Create Gpu Instance
    nvidia-smi mig -i ${DEVICE_ID} -cgi "${gi[${num_procs}]}"

    # Get GPU Instance ID
    gi_id_arr=($(nvidia-smi mig -i ${DEVICE_ID} -lgi | awk '{print $6}' | grep -P '^\d+$'))

    # Create Compute Instance
    for gi_id in "${gi_id_arr[@]}"
    do
        # Choose the largest sub-chunk
        ci=$(nvidia-smi mig -i ${DEVICE_ID} -gi ${gi_id} -lcip | grep "\*" | awk '{print $6}' | tr -d "*")
        nvidia-smi mig -i ${DEVICE_ID} -gi ${gi_id} -cci ${ci}
    done
}

function cleanup_mig_if_needed()
{
    mode=$1
    if [[ ${mode} != "mig" ]]; then
        return
    fi

	# Delete the GPU profile
	exit_code=0
	while :;
	do
		# Cleanup Compute Instances
		ci_arr=($(nvidia-smi mig -lci | awk '{print $2}' | grep -P '^\d+$'))
		gi_arr=($(nvidia-smi mig -lci | awk '{print $3}' | grep -P '^\d+$'))
		i=0
		while [[ $i -lt ${#gi_arr[*]} ]];
		do
			ci=${ci_arr[$i]}
			gi=${gi_arr[$i]}
			nvidia-smi mig -dci -ci ${ci} -gi ${gi}
			i=$(( $i + 1))
		done

		# Cleanup GPU Instances
		nvidia-smi mig -i ${DEVICE_ID} -dgi
		exit_code=$?
		if [[ ${exit_code} -eq 0 || ${exit_code} -eq 6 ]]; then
			break
		fi
		print "nvidia-smi mig -i ${DEVICE_ID} -dgi failed. Trying in 1s"
		sleep 1
	done
}

function cleanup()
{
    mode=$1
    disable_mps_if_needed ${mode}
    cleanup_mig_if_needed ${mode}
}

function calc_mean_sd() {
    arr=("$@")
    mean=$(echo ${arr[@]} | awk '{for(i=1;i<=NF;i++){sum+=$i};print sum/NF}')
    sd=$(echo ${arr[@]} | awk -vM=$mean '{for(i=1;i<=NF;i++){sum+=($i-M)*($i-M)};print sqrt(sum/NF)}')
    echo "$mean, $sd"
}

function calc_min_max_mean {
    arr=("$@")
    max=$(echo ${arr[@]} | awk -vmax=${arr[0]} '{for(i=1;i<=NF;i++){(( $i > max )) && max=$i};print max}')
    min=$(echo ${arr[@]} | awk -vmin=${arr[0]} '{for(i=1;i<=NF;i++){(( $i < min )) && min=$i};print min}')
    mean=$(echo ${arr[@]} | awk '{for(i=1;i<=NF;i++){sum+=$i};print sum/NF}')
    echo "$min, $max, $mean"
}

function wait_till_one_process_exits {
    pids=("$@")
    cpu=0
    done=0
    while true; do
        for pid in ${pids[@]}; do
            if ! taskset -c ${cpu} kill -0 $pid >/dev/null 2>&1; then
                done=1
                break
            fi
        done

        if [[ ${done} -eq 1 ]]; then
            echo "Sending SIGUSR2 to ${pids[@]}"
            kill -SIGUSR2 ${pids[@]} >/dev/null 2>&1
            break
        fi
	done

    for pid in ${pids[@]}; do
        echo "Waiting for ${pid} to finish"
        while taskset -c 0 kill -0 ${pid} >/dev/null 2>&1; do sleep 1; done
    done
}

mps_mig_percentages=("" "100" "57,43" "42,29,29" "29,29,28,14" "29,29,14,14,14" "29,15,14,14,14,14" "15,15,14,14,14,14,14")
mps_equi_percentages=("" "100" "50,50" "34,33,33" "25,25,25,25" "20,20,20,20,20" "17,17,17,17,16,16" "15,15,14,14,14,14,14")
mps_chunks_percentages=("" "14" "14,14" "14,14,14" "14,14,14,14" "14,14,14,14,14" "14,14,14,14,14,14" "14,14,14,14,14,14,14")
attempts=10
# Command to profile
# nsys profile --stats=true --force-overwrite true --wait=all -o trial
DEVICE_ID=0
nvidia-smi -i ${DEVICE_ID} -pm ENABLED
