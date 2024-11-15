#!/bin/bash

BASE_DIR="../data/Mistralv03_CS_alpha08/Mistralv03-alpha08"
SUBFOLDERS=("wikitext" "wikinews" "book")

for subfolder in "${SUBFOLDERS[@]}"; do
    for file in "$BASE_DIR/$subfolder"/*.json; do
        if [[ -f "$file" && ! "$file" =~ .*result\.json$ ]]; then
            echo "Processing file: $file"
            CUDA_VISIBLE_DEVICES=0 python ../compute_coherence.py \
                --opt_model_name facebook/opt-2.7b \
                --test_path "$file"
        fi
    done
done