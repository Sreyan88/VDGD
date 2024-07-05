import argparse
import torch
import os
import json
from tqdm import tqdm
import copy
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, DEFAULT_IMAGE_PATCH_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from PIL import Image
import numpy as np

from transformers import set_seed
import pickle

llm_templates = {
    "lmsys/vicuna-7b-v1.5": "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: {in_text} ASSISTANT:",
    "meta-llama/Llama-2-7b-chat-hf": "[INST] <<SYS>>\nYou are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.\n<</SYS>>\n\n{in_text} [/INST]"
}

final_prompt = "The input image is described as follows: {desc} {question}"

def greedy_decoding(model, tokenizer, image_processor, prompt, image_path, do_sample=False, max_len=512, epsilon=0.001, alpha=0.1):

    bos_token_id = tokenizer.bos_token_id
    eos_token_id = tokenizer.eos_token_id
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id else eos_token_id

    device = model.device

    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(device)
    image = Image.open(image_path)
    image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0].unsqueeze(0).half().to(device)
    input_token_len = input_ids.shape[1]
    unfinished_sequences = torch.ones(input_ids.shape[0], dtype=torch.long, device=input_ids.device)
    eos_token_id_tensor = torch.tensor([eos_token_id]).to(input_ids.device) if eos_token_id is not None else None

    for _ in range(max_len):
        greedy_output = model.generate(input_ids, images=image_tensor, max_new_tokens=1, do_sample=False, return_dict_in_generate=True, output_scores=True, output_logits=True, bos_token_id=bos_token_id, eos_token_id=eos_token_id, pad_token_id=eos_token_id)
        logit = greedy_output['logits'][0]

        x = nn.functional.softmax(logit, dim=-1)

        next_id = torch.argmax(x)
        next_id = next_id * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)
        
        unfinished_sequences = unfinished_sequences.mul(
                        next_id.tile(eos_token_id_tensor.shape[0], 1).ne(eos_token_id_tensor.unsqueeze(1)).prod(dim=0)
                    )
    
        input_ids = torch.cat([input_ids, next_id[:, None]], dim=-1)
        # print(tokenizer.batch_decode(input_ids[:, input_token_len:], skip_special_tokens=True)[0])
        # print()
        if unfinished_sequences.max() == 0:
            break

    generated_results = tokenizer.batch_decode(input_ids[:, input_token_len:], skip_special_tokens=True)[0]
    return generated_results

def sample_decoding(model, tokenizer, image_processor, prompt, image_path, do_sample=False, max_len=512, epsilon=0.001, alpha=0.1):

    bos_token_id = tokenizer.bos_token_id
    eos_token_id = tokenizer.eos_token_id
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id else eos_token_id

    device = model.device

    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(device)
    image = Image.open(image_path)
    image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0].unsqueeze(0).half().to(device)
    input_token_len = input_ids.shape[1]

    unfinished_sequences = torch.ones(input_ids.shape[0], dtype=torch.long, device=input_ids.device)
    eos_token_id_tensor = torch.tensor([eos_token_id]).to(input_ids.device) if eos_token_id is not None else None

    for _ in range(max_len):
        output = model.generate(input_ids, images=image_tensor, max_new_tokens=1, do_sample=True, top_p=0.5, temperature=0.5, return_dict_in_generate=True, output_scores=True, output_logits=True, bos_token_id=bos_token_id, eos_token_id=eos_token_id, pad_token_id=eos_token_id)
        logit = output['scores'][0]

        x = nn.functional.softmax(logit, dim=-1)
        next_id = torch.multinomial(x, num_samples=1).squeeze(1)
        next_id = next_id * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)
        
        unfinished_sequences = unfinished_sequences.mul(
                        next_id.tile(eos_token_id_tensor.shape[0], 1).ne(eos_token_id_tensor.unsqueeze(1)).prod(dim=0)
                    )
    
        input_ids = torch.cat([input_ids, next_id[:, None]], dim=-1)
        # print(tokenizer.batch_decode(input_ids[:, input_token_len:], skip_special_tokens=True)[0])
        # print()
        if unfinished_sequences.max() == 0:
            break

    generated_results = tokenizer.batch_decode(input_ids[:, input_token_len:], skip_special_tokens=True)[0]
    return generated_results

def top_p_and_temp_filtering(logit, top_p=0.5, temperature=0.5):
    logits_processed = logit / temperature
    sorted_logits, sorted_indices = torch.sort(logits_processed, descending=False)
    cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)

    # Remove tokens with cumulative top_p above the threshold (token with 0 are kept)
    sorted_indices_to_remove = cumulative_probs <= (1 - top_p)
    # Keep at least min_tokens_to_keep
    sorted_indices_to_remove[..., -1 :] = 0

    # scatter sorted tensors to original indexing
    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
    logits_processed = logits_processed.masked_fill(indices_to_remove, float("-inf"))
    return logits_processed
    
def vdgd(model, tokenizer, image_processor, prompt, image_path, log_tup, do_sample=False, max_len=512, epsilon=0.001, alpha=0.1):

    bos_token_id = tokenizer.bos_token_id
    eos_token_id = tokenizer.eos_token_id
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id else eos_token_id

    device = model.device

    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(model.device)
    image = Image.open(image_path)
    image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0].unsqueeze(0).half().to(model.device)
    input_token_len = input_ids.shape[1]

    one_hot = torch.eye(len(tokenizer)).to(model.device)

    unfinished_sequences = torch.ones(input_ids.shape[0], dtype=torch.long, device=input_ids.device)
    eos_token_id_tensor = torch.tensor([eos_token_id]).to(input_ids.device) if eos_token_id is not None else None
    for _ in range(max_len):
        output = model.generate(input_ids, images=image_tensor, max_new_tokens=1, do_sample=False, return_dict_in_generate=True, output_scores=True, output_logits=True, bos_token_id=bos_token_id, eos_token_id=eos_token_id, pad_token_id=eos_token_id)

        logit = output['logits'][0]
        
        logit = top_p_and_temp_filtering(logit)     

        top_k_logit_idxs = (logit[0]!=-float("inf")).nonzero()

        for tpo_k_idx in top_k_logit_idxs:
            if args.kl_reduction == "min":
                kl_div = float("inf")
                for j in range(len(log_tup)):
                    kl_div = min(kl_div, torch.nn.functional.kl_div(log_tup[j], one_hot[tpo_k_idx.item()], reduction='batchmean').item())
                logit[0][tpo_k_idx.item()] = -kl_div
            elif args.kl_reduction == "avg":
                kl_div = 0.0
                for j in range(len(log_tup)):
                    kl_div += torch.nn.functional.kl_div(log_tup[j], one_hot[tpo_k_idx.item()], reduction='batchmean').item()
                logit[0][tpo_k_idx.item()] = -(kl_div/len(log_tup))
            elif args.kl_reduction == "sum":
                kl_div = 0.0
                for j in range(len(log_tup)):
                    kl_div += torch.nn.functional.kl_div(log_tup[j], one_hot[tpo_k_idx.item()], reduction='batchmean').item()
                logit[0][tpo_k_idx.item()] = -(kl_div)      

        x = nn.functional.softmax(logit, dim=-1)

        next_id = torch.multinomial(x, num_samples=1).squeeze(1)        
        next_id = next_id * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)
        
        unfinished_sequences = unfinished_sequences.mul(
                        next_id.tile(eos_token_id_tensor.shape[0], 1).ne(eos_token_id_tensor.unsqueeze(1)).prod(dim=0)
                    )
    
        input_ids = torch.cat([input_ids, next_id[:, None]], dim=-1)

        if unfinished_sequences.max() == 0:
            break

    generated_results = tokenizer.batch_decode(input_ids[:, input_token_len:], skip_special_tokens=True)[0]
    return generated_results

def eval_model(args):
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    compute_dtype = torch.float16

    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)

    questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]
    descriptions = [json.loads(q) for q in open(os.path.expanduser(args.desc_file), "r")]
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "a")

    with open(args.logits_file, 'rb') as f:
        logit_tuples = pickle.load(f)

    for line, desc, log_tup in tqdm(zip(questions, descriptions, logit_tuples)):
        log_tup = tuple(torch.nn.functional.log_softmax(tensor[0].to(model.device), dim=-1) for tensor in log_tup)
        idx = line["id"]
        image_file = line["image"]
        for conv in line["conversations"]:
            if conv["from"] == "human":
                qs = final_prompt.format(desc=desc["text"], question=conv["value"])
                break
        cur_prompt = qs

        conv = conv_templates[args.conv_mode].copy()
        
        if 'POPE' in args.question_file:
            conv.append_message(conv.roles[0], qs + " Please answer this question with one word.")
            conv.append_message(conv.roles[1], None)
        else:
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        if args.decoding_type == "vdgd":
            results = vdgd(model, tokenizer, image_processor, prompt, os.path.join(args.image_folder, image_file), log_tup)
        elif args.decoding_type == "sd":
            results = sample_decoding(model, tokenizer, image_processor, prompt, os.path.join(args.image_folder, image_file))
        elif args.decoding_type == "gd":
            results = greedy_decoding(model, tokenizer, image_processor, prompt, os.path.join(args.image_folder, image_file))

        ans_file.write(json.dumps({"question_id": idx,
                                   "prompt": cur_prompt,
                                   "text": results,
                                   "model_id": model_name,
                                   "image": image_file,
                                   "metadata": {}}) + "\n")
        ans_file.flush()
    ans_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model_base", type=str, default=None)
    parser.add_argument("--image_folder", type=str, default="")
    parser.add_argument("--question_file", type=str, default="./vallu_benchmark.jsonl")
    parser.add_argument("--answers_file", type=str, default="./out.jsonl")
    parser.add_argument("--desc_file", type=str, default="./desc.jsonl")
    parser.add_argument("--logits_file", type=str, required=True)
    parser.add_argument("--decoding_type", type=str, required=True)
    parser.add_argument("--kl_reduction", type=str, default="min")
    parser.add_argument("--conv_mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--noise_step", type=int, default=500)
    parser.add_argument("--use_cd", action='store_true', default=False)
    parser.add_argument("--cd_alpha", type=float, default=1)
    parser.add_argument("--cd_beta", type=float, default=0.1)
    parser.add_argument("--use_dd", action='store_true', default=False)
    parser.add_argument("--use_ori", action='store_true', default=False)
    parser.add_argument("--use_dd_unk", action='store_true', default=False)
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()
    
    set_seed(args.seed)
    answers_file = copy.deepcopy(args.answers_file)
    args.temperature = 0
    args.top_p = None
    args.top_k = None
    args.do_sample = False
    default_args = copy.deepcopy(args)

    args.answers_file = answers_file.replace('setting', 'default')
    eval_model(args)