#!/bin/bash

COLING_DATA_DIR="."

# Specify the subfolders you want to process
subfolders=("book" "wikinews" "wikitext")

# Find all immediate subdirectories of COLING_DATA_DIR
for BASE_DIR in "$COLING_DATA_DIR"/{Deepseek,Falcon}*/ ; do
    if [ ! -d "$BASE_DIR" ]; then
        continue  # Skip if not a directory (in case no matches were found)
    fi
    BASE_DIR=${BASE_DIR%*/}  # Remove trailing slash
    echo "Processing directory: $BASE_DIR"

    for subfolder in "${subfolders[@]}"; do
        # Search for the subfolder in the current directory and one level deeper
        found_dirs=$(find "$BASE_DIR" -maxdepth 2 -type d -name "$subfolder")
        
        if [ -n "$found_dirs" ]; then
            for dir in $found_dirs; do
                echo "Processing subfolder: $dir"
                for file in "$dir"/*.json; do
                    if [[ -f "$file" && ! "$file" =~ .*result\.json$ ]]; then
                        echo "Processing file: $file"
                        
                        # Measure MAUVE
                        echo "Measuring MAUVE..."
                        CUDA_VISIBLE_DEVICES=0 python ../compute_coherence.py \
                            --opt_model_name facebook/opt-2.7b \
                            --test_path "$file"

                        echo "------------------------"
                    fi
                done
            done
        else
            echo "Subfolder '$subfolder' not found in $BASE_DIR or its immediate subdirectories"
        fi
    done
done