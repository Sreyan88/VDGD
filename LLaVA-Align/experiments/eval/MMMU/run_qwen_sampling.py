import torch
import os
import random
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


import numpy as np
from tqdm import tqdm
from PIL import Image

from datasets import load_dataset, concatenate_datasets
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path

from argparse import ArgumentParser

from transformers import set_seed,AutoTokenizer,AutoModelForCausalLM
from Qwen_VL.modeling_qwen import QWenLMHeadModel
from utils.data_utils import load_yaml, construct_prompt, save_json, process_single_sample, CAT_SHORT2LONG
from utils.model_utils import call_llava_engine_df, llava_image_processor
from utils.eval_utils import parse_multi_choice_response, parse_open_response

def call_qwen_engine_df(args, sample, model, tokenizer=None, processor=None):

    prompt = sample['final_input_prompt']
    question = '<img> /myfolder/ image </img>{} Answer:'.format(prompt.replace('<image 1>', ''))
    
    input_ids = tokenizer([question], return_tensors='pt', padding='longest')

    image = sample['image'].convert("RGB")
    image = model.transformer.visual.image_transform(image).unsqueeze(0).to(model.device)

    if image is not None:
        output_ids = model.generate(
            input_ids=input_ids.input_ids.cuda(),
            attention_mask=input_ids.attention_mask.cuda(),
            images = image,
            do_sample=True,
            max_new_tokens=20,
            min_new_tokens=1,
            length_penalty=1,
            num_return_sequences=1,
            output_hidden_states=True,
            use_cache=True,
            pad_token_id=tokenizer.eod_id,
            eos_token_id=tokenizer.eod_id,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
        )

        response = [
            tokenizer.decode(_[input_ids.input_ids.size(1):].cpu(),
                             skip_special_tokens=True).strip() for _ in output_ids
        ][0]
        # response = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
    else:  # multiple images actually
        if sample['question_type'] == 'multiple-choice':
            all_choices = sample['all_choices']
            response = random.choice(all_choices)
        else:
            response = 'INVALID GENERATION FOR MULTIPLE IMAGE INPUTS'

    return response


def run_model(args, samples, model, call_model_engine_fn=None, tokenizer=None, processor=None):
    out_samples = dict()
    with torch.no_grad():
        for sample in tqdm(samples):
            response = call_model_engine_fn(args, sample, model, tokenizer, processor)

            if sample['question_type'] == 'multiple-choice':
                pred_ans = parse_multi_choice_response(response, sample['all_choices'], sample['index2ans'])
            else:  # open question
                pred_ans = response
            out_samples[sample['id']] = pred_ans
    return out_samples

def set_seed(seed_value):
    """
    Set the seed for PyTorch (both CPU and CUDA), Python, and NumPy for reproducible results.

    :param seed_value: An integer value to be used as the seed.
    """
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)  # For multi-GPU setups
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main():
    parser = ArgumentParser()
    parser.add_argument('--output_path', type=str, default='eval/MMMU/outputs/qwen_sampling_test.json',
                        help='name of saved json')
    parser.add_argument('--config_path', type=str, default="eval/MMMU/configs/llava1.5.yaml")
    parser.add_argument('--data_path', type=str, default="MMMU/MMMU") # hf dataset path.
    parser.add_argument('--model_path', type=str, default="Qwen/Qwen-VL")
    parser.add_argument('--split', type=str, default='validation')
    parser.add_argument('--seed', type=int, default=42)

    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--top_k", type=int, default=None)

    args = parser.parse_args()
    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
    set_seed(args.seed)

    print('llava_initializing...')
    processor = None
    call_model_engine = call_qwen_engine_df
    vis_process_func = llava_image_processor

    # load config and process to one value
    args.config = load_yaml(args.config_path)
    for key, value in args.config.items():
        if key != 'eval_params' and type(value) == list:
            assert len(value) == 1, 'key {} has more than one value'.format(key)
            args.config[key] = value[0]

    # run for each subject
    sub_dataset_list = []
    for subject in CAT_SHORT2LONG.values():
        sub_dataset = load_dataset(args.data_path, subject, split=args.split)
        sub_dataset_list.append(sub_dataset)

    # merge all dataset
    dataset = concatenate_datasets(sub_dataset_list)

    model_path = os.path.expanduser(args.model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tokenizer.padding_side = 'left'
    tokenizer.pad_token_id = tokenizer.eod_id
    model = QWenLMHeadModel.from_pretrained(
        model_path,
        device_map="cuda",
        trust_remote_code=True
    ).eval()

    samples = []
    for sample in dataset:
        sample = process_single_sample(sample)

        sample = construct_prompt(sample, args.config)
        samples.append(sample)

    import copy
    default_args = copy.deepcopy(args)
    output_path = copy.deepcopy(args.output_path)

    args.do_sample = True
    args.temperature = 1.0
    args.output_path = output_path.replace('setting', f'default')
    out_samples = run_model(args, samples, model, call_model_engine, tokenizer, processor)
    save_json(args.output_path, out_samples)

    args = default_args

    for temp in np.arange(0.05, 1.05, 0.05):
        temp = np.round(temp, 2)
        print(f"Running temp = {temp}")
        
        args.do_sample = True
        args.temperature = temp
        args.output_path = output_path.replace('setting', f'temp_{temp}')
        
        out_samples = run_model(args, samples, model, call_model_engine, tokenizer, processor)
        save_json(args.output_path, out_samples)

    args = default_args
            
    for top_p in np.arange(0, 1.05, 0.05):
        top_p = np.round(top_p, 2)
        print(f"Running top_p = {top_p}")
        
        args.do_sample = True
        args.top_p=top_p
        args.output_path = output_path.replace('setting', f'top_p_{top_p}')
        
        out_samples = run_model(args, samples, model, call_model_engine, tokenizer, processor)
        save_json(args.output_path, out_samples)
    args = default_args
    
    for top_k in [1, 2, 5, 10, 20, 50, 100, 200, 500]:
        print(f"Running top_k = {top_k}")
        
        args.do_sample = True
        args.top_k = top_k
        args.output_path = output_path.replace('setting', f'top_k_{top_k}')
        
        out_samples = run_model(args, samples, model, call_model_engine, tokenizer, processor)
        save_json(args.output_path, out_samples)
    
    # metric_dict.update({"num_example": len(out_samples)})
    # save_json(save_result_path, metric_dict)


if __name__ == '__main__':
    main()

