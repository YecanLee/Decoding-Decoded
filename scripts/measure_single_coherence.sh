#!/bin/bash

TEST_PATH=${1:-"../data/Qwen_beam/Qwen2-beam/wikinews/wikinews_qwen2_num_beams_50.json"}

CUDA_VISIBLE_DEVICES=0 python ../compute_golden_coherence.py \
    --opt_model_name facebook/opt-2.7b \
    --test_path "$TEST_PATH"
