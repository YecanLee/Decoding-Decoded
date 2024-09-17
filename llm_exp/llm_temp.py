import os 
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
sys.path.append(root_dir)

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import numpy as np
import os
import json
from tqdm import trange
from helpers.utils import load_data

# A fast inference setting for Ampere GPUs
if torch.cuda.get_device_properties(0).major >= 8:
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    print('Fast inference setting for Ampere GPUs is enabled 🔥🔥🔥.')

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default="meta-llama/Meta-Llama-3.1-8B")
    parser.add_argument('--dataset_prefix', type=str, default='./data')
    parser.add_argument('--dataset', type=str, default='wikitext')
    parser.add_argument('--save_path_prefix', type=str, default='Llama3_1-temp')
    parser.add_argument('--cuda', type=int, default=0)
    parser.add_argument('--temp', required=True, type=float)
    parser.add_argument('--save_file', required=True, type=str)
    args = parser.parse_args()
    if torch.cuda.is_available():
        print('Cuda is available.')
    cuda_available = torch.cuda.is_available()
    device = torch.device(f'cuda:{args.cuda}' if cuda_available else 'cpu')

    assert args.dataset in ['book', 'wikinews', 'wikitext'], "Dataset must be one of 'book', 'wikinews', or 'wikitext'"
    full_data_path = f'{args.dataset_prefix}/{args.dataset}_contrastive_gpt2-xl_256.jsonl'
    print(f'Full data path is {full_data_path}')

    save_path_prefix = f'{args.save_path_prefix}/{args.dataset}/'
    print(f"Save path prefix is {save_path_prefix}")
    if not os.path.exists(save_path_prefix):
        os.makedirs(save_path_prefix, exist_ok=True)
    save_name = f'{args.dataset}_{args.save_file}_temp_{args.temp}.json'
    save_path = os.path.join(save_path_prefix, save_name)
    print(f'Result saving path is {save_path}')

    print('Loading model... 🔧🔧🔧')
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, fast=True)
    model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.bfloat16, device_map="cpu")
    model = torch.compile(model, mode="max-autotune")
    model.to(device)

    prefix_text_list, prefix_token_id_list, reference_text_list = load_data(full_data_path, tokenizer, mode=args.dataset)

    print('Performing inference... 🚀🚀🚀')
    data_num = len(prefix_text_list)
    print(data_num)
    result_list = []

    with torch.inference_mode():
        for index in trange(data_num, desc='Inferring... ⌛⌛⌛'):
            one_prefix_text = prefix_text_list[index]
            one_reference_text = reference_text_list[index]
            model_inputs = tokenizer([one_prefix_text], return_tensors="pt").to(device)
            _, prefix_len = model_inputs.input_ids.size()
            generated_ids = model.generate(**model_inputs, do_sample=True, temperature=args.temp, max_new_tokens=256, pad_token_id=tokenizer.eos_token_id, top_k=0)
            one_generation_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]    
                    
            one_res_dict = {
                'prefix_text': one_prefix_text,
                'reference_text': one_reference_text,
                'generated_result': {
                    '0': one_generation_text
                }
            }
            result_list.append(one_res_dict)
        print('Inference completed! 🎉🎉🎉')

        with open(save_path, 'w') as outfile:
            json.dump(result_list, outfile, indent=4)


"""
python llm_exp/llama_temp.py \
--temp 0.1 \
--save_file llama3 \
--dataset book \
--model_name meta-llama/Meta-Llama-3.1-8B \
--save_path_prefix Llama-3_1-temp \
--cuda 0 \
&&
python llm_exp/llama_temp.py \
--temp 0.3 \
--save_file llama3 \
--dataset book \
--model_name meta-llama/Meta-Llama-3.1-8B \
--save_path_prefix Llama-3_1-temp \
--cuda 0 \
&&
python llm_exp/llama_temp.py \
--temp 0.5 \
--save_file llama3 \
--dataset book \
--model_name meta-llama/Meta-Llama-3.1-8B \
--save_path_prefix Llama-3_1-temp \
--cuda 0 \
&&
python llm_exp/llama_temp.py \
--temp 0.7 \
--save_file llama3 \
--dataset book \
--model_name meta-llama/Meta-Llama-3.1-8B \
--save_path_prefix Llama-3_1-temp \
--cuda 0 \
&&
python llm_exp/llama_temp.py \
--temp 0.9 \
--save_file llama3 \
--dataset book \
--model_name meta-llama/Meta-Llama-3.1-8B \
--save_path_prefix Llama-3_1-temp \
--cuda 0 \
&&
python llm_exp/llama_temp.py \
--temp 1.0 \
--save_file llama3 \
--dataset book \
--model_name meta-llama/Meta-Llama-3.1-8B \
--save_path_prefix Llama-3_1-temp \
--cuda 0 \
"""