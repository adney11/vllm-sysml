#!/bin/bash -x
psudo() { sudo env PATH="$PATH" "$@"; }
for input_tokens in 128 256 512 1024 2048 4096 8192;
do
    for output_tokens in 128 256 512 1024 2048 4096 8192;
    do
        psudo HF_HOME=/sysml/.cache TMPDIR=/sysml/tmp python sysml_tests/llm_engine_example_profile.py --max-num-seqs $input_tokens --max-paddings $output_tokens > /sysml/misc/asad/SysML-splitwiser/repos/vllm-sysml/profiling_results/no_ncu/input$input_tokens-output$output_tokens.txt
    done
done