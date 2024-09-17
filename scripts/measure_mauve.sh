#!/bin/bash

BASE_DIR="../coling_data/Mistralv03_CS_alpha08/Mistralv03-alpha08/"
SUBFOLDERS=("wikitext" "wikinews" "book")

for subfolder in "${SUBFOLDERS[@]}"; do
    for json_file in "$BASE_DIR/$subfolder"/*.json; do
        if [[ -f "$json_file" && ! "$json_file" =~ .*result\.json$ ]]; then
            echo "Processing: $json_file"
            CUDA_VISIBLE_DEVICES=0 python ../measure_diversity_mauve_gen_length.py \
                --test_path "$json_file"
            echo "------------------------"
        fi
    done
done