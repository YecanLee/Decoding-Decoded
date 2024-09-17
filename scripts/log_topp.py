import os
import json
import re
import argparse

def extract_model_info(folder_path):
    parts = folder_path.split(os.path.sep)
    model_name = parts[-2].split('-')[0]  # Extracts 'Llama3' from 'Llama-3_1-topp'
    decoding_strategy = "TopP"
    return model_name, decoding_strategy

def log_metrics_results(folder_path, save_path):
    output_file = os.path.join(folder_path, f"{save_path}_metrics_results_log.txt")
    model_name, decoding_strategy = extract_model_info(folder_path)

    coherence_pattern = r"(.+)_(.+)_p_(.+)_opt-2\.7b_coherence_result\.json"
    diversity_pattern = r"(.+)_(.+)_p_(.+)_diversity_mauve_gen_length_result\.json"
    
    results = {}

    with open(output_file, 'w') as log_file:
        log_file.write("Dataset | Model | Decoding_Strategy | P | Coherence Mean | Prediction Diversity Mean | MAUVE Mean\n")
        log_file.write("-" * 140 + "\n")
        
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            
            coherence_match = re.match(coherence_pattern, filename)
            diversity_match = re.match(diversity_pattern, filename)
            
            if coherence_match or diversity_match:
                match = coherence_match or diversity_match
                dataset, _, p = match.groups()
                key = (dataset, p)
                
                if key not in results:
                    results[key] = {"coherence_mean": "N/A", "pred_div_mean": "N/A", "mauve_mean": "N/A"}
                
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    if isinstance(data, list) and len(data) > 0:
                        if coherence_match:
                            results[key]["coherence_mean"] = data[0].get("coherence_mean", "N/A")
                        elif diversity_match:
                            results[key]["pred_div_mean"] = data[0].get("diversity_dict", {}).get("prediction_div_mean", "N/A")
                            results[key]["mauve_mean"] = data[0].get("mauve_dict", {}).get("mauve_mean", "N/A")
        
        for (dataset, p), metrics in results.items():
            log_file.write(f"{dataset} | {model_name} | {decoding_strategy} | {p} | {metrics['coherence_mean']} | {metrics['pred_div_mean']} | {metrics['mauve_mean']}\n")
    
    print(f"Results have been logged to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Log metrics results from JSON files in a folder.")
    parser.add_argument("folder_path", type=str, help="Path to the folder containing the JSON files.")
    parser.add_argument("save_path", type=str, help="Name for the output log file (without extension).")
    args = parser.parse_args()

    log_metrics_results(args.folder_path, args.save_path)