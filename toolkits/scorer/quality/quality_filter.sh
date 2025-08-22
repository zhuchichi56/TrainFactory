#!/bin/bash

export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

DEFAULT_CUDA_DEVICES="0,1,2,3,4,5,6,7"
DEVICES=$DEFAULT_CUDA_DEVICES


while [[ $# -gt 0 ]]; do
    case $1 in
        --input_file)
            INPUT_FILE="$2"
            shift 2
            ;;
        --result_file)
            RESULT_FILE="$2"
            shift 2
            ;;
        --max_length)
            MAX_LENGTH="$2"
            shift 2
            ;;
        --batch_size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --need_filtering)
            NEED_FILTERING="$2"
            shift 2
            ;;
        --threshold)
            THRESHOLD="$2"
            shift 2
            ;;
        --devices)
            DEVICES="$2"
            shift 2
            ;;
        *)
            shift
            ;;
    esac
done

# 使用 CUDA_VISIBLE_DEVICES 指定 GPU
CUDA_VISIBLE_DEVICES=$DEVICES accelerate launch --main_process_port 29999 data_selection/quality_filter.py \
    --input_file "$INPUT_FILE" \
    --result_file "$RESULT_FILE" \
    --max_length "$MAX_LENGTH" \
    --batch_size "$BATCH_SIZE" \
    --need_filtering "$NEED_FILTERING" \
    --threshold "$THRESHOLD"






