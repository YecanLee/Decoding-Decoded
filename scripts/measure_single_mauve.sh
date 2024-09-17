#!/bin/bash

TEST_PATH=${1:-"../data/Qwen_beam/Qwen2-beam/wikinews/wikinews_qwen2_num_beams_50.json"}

CUDA_VISIBLE_DEVICES=0 python ../measure_diversity_mauve_gen_length.py\
    --test_path "$TEST_PATH"
