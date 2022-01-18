#!/bin/bash


CONDA_ACTIVATE='conda activate py37'
CD_TO_DIR='cd ~/code/gambit/bench'

# setenv TF_CPP_MIN_LOG_LEVEL 2 \; \

function run_tmux_session_gpu() {
    device_order="PCI_BUS_ID"
    gpu_index=$1
    cmd=$2
    tmux new-session -s "gpu${gpu_index}" -d \; \
        setenv CUDA_DEVICE_ORDER $device_order \; \
        setenv CUDA_VISIBLE_DEVICES $gpu_index \; \
        send-keys -t 0 "export CUDA_DEVICE_ORDER="$device_order c-m \; \
        send-keys -t 0 "export CUDA_VISIBLE_DEVICES="$gpu_index C-m \; \
        send-keys -t 0 'echo $CUDA_DEVICE_ORDER' C-m \; \
        send-keys -t 0 'echo $CUDA_VISIBLE_DEVICES' C-m \; \
        send-keys -t 0 'echo $TF_CPP_MIN_LOG_LEVEL' C-m \; \
        send-keys -t 0 "${CONDA_ACTIVATE}" C-m \; \
        send-keys -t 0 "${CD_TO_DIR}" C-m \;
    echo "Created tmux session for gpu${gpu_index} and command ${cmd}"
}


function run_tmux_session_cpu() {
    index=$1
    cmd=$2
    tmux new-session -s "cpu${index}" -d \; \
        setenv CUDA_VISIBLE_DEVICES '' \; \
        send-keys -t 0 "export CUDA_VISIBLE_DEVICES=''" C-m \; \
        send-keys -t 0 'echo $CUDA_VISIBLE_DEVICES' C-m \; \
        send-keys -t 0 "${CONDA_ACTIVATE}" C-m \; \
        send-keys -t 0 "${CD_TO_DIR}" C-m \;
    echo "Created tmux session for cpu${index} and command ${cmd}";
}


run_tmux_session_gpu '0';
run_tmux_session_gpu '1';
run_tmux_session_gpu '2';
run_tmux_session_gpu '3';
# run_tmux_session_gpu '3';
# run_tmux_session_gpu '2';
# run_tmux_session_gpu '3';

tmux ls
