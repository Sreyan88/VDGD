"""
This is a demo for using CogAgent and CogVLM in CLI
Make sure you have installed vicuna-7b-v1.5 tokenizer model (https://huggingface.co/lmsys/vicuna-7b-v1.5), full checkpoint of vicuna-7b-v1.5 LLM is not required.
In this demo, We us chat template, you can use others to replace such as 'vqa'.
Strongly suggest to use GPU with bfloat16 support, otherwise, it will be slow.
Mention that only one picture can be processed at one conversation, which means you can not replace or insert another picture during the conversation.
"""
import json

import argparse
import torch
from tqdm import tqdm
import sys
import os
from PIL import Image
from transformers import AutoModelForCausalLM, LlamaTokenizer

parser = argparse.ArgumentParser()
parser.add_argument("--quant", choices=[4], type=int, default=None, help='quantization bits')
parser.add_argument("--from_pretrained", type=str, default="THUDM/cogvlm-chat-hf", help='pretrained ckpt')
parser.add_argument("--local_tokenizer", type=str, default="lmsys/vicuna-7b-v1.5", help='tokenizer path')
parser.add_argument("--fp16", action="store_true")
parser.add_argument("--bf16", action="store_true")
parser.add_argument("--file_name", type=str, required=True)
parser.add_argument("--out_file_name", type=str, required=True)

args = parser.parse_args()

file_name = args.file_name
out_file_name = args.out_file_name

MODEL_PATH = args.from_pretrained
TOKENIZER_PATH = args.local_tokenizer
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

current_dir = os.path.dirname(os.path.abspath(__file__))
datasets_dir = os.path.abspath(os.path.join(current_dir, '..', 'datasets'))
align_tds_dir = os.path.abspath(os.path.join(current_dir, '..', 'AlignTDS'))
inference_gen_dir = os.path.abspath(os.path.join(current_dir, '..', 'inference_generations'))

tokenizer = LlamaTokenizer.from_pretrained(TOKENIZER_PATH)
if args.bf16:
    torch_type = torch.bfloat16
else:
    torch_type = torch.float16

print("========Use torch type as:{} with device:{}========\n\n".format(torch_type, DEVICE))

if args.quant:
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch_type,
        low_cpu_mem_usage=True,
        load_in_4bit=True,
        trust_remote_code=True
    ).eval()
else:
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch_type,
        low_cpu_mem_usage=True,
        load_in_4bit=args.quant is not None,
        trust_remote_code=True
    ).to(DEVICE).eval()

text_only_template = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: {} ASSISTANT:"

file = open(os.path.join(datasets_dir, f'{file_name}.jsonl'), 'r')
file = file.readlines()
file = [eval(i) for i in file]

align_data = []
results = []
for idx, i in tqdm(enumerate(file)):
    image_path = os.path.join(datasets_dir, i['image'])
    image = Image.open(image_path).convert('RGB')

    history = []

    queries = i['conversations']
    query = ""
    for k in queries:
        if k['from'] == 'human':
            query += k['value'].replace("<image>","")
    pure_query = query
    query = text_only_template.format(query)

    if image is None:
        continue
        if text_only_first_query:
            query = text_only_template.format(query)
            text_only_first_query = False
        else:
            old_prompt = ''
            for _, (old_query, response) in enumerate(history):
                old_prompt += old_query + " " + response + "\n"
            query = old_prompt + "USER: {} ASSISTANT:".format(query)

    if image is None:
        input_by_model = model.build_conversation_input_ids(tokenizer, query=query, history=history, template_version='base')
    else:
        input_by_model = model.build_conversation_input_ids(tokenizer, query=query, history=history, images=[image])

    inputs = {
        'input_ids': input_by_model['input_ids'].unsqueeze(0).to(DEVICE),
        'token_type_ids': input_by_model['token_type_ids'].unsqueeze(0).to(DEVICE),
        'attention_mask': input_by_model['attention_mask'].unsqueeze(0).to(DEVICE),
        'images': [[input_by_model['images'][0].to(DEVICE).to(torch_type)]] if image is not None else None,
    }
    if 'cross_images' in input_by_model and input_by_model['cross_images']:
        inputs['cross_images'] = [[input_by_model['cross_images'][0].to(DEVICE).to(torch_type)]]

    gen_kwargs = {"max_new_tokens": 512,
                    "do_sample": False, "num_beams": 1}
    with torch.no_grad():
        outputs = model.generate(**inputs, **gen_kwargs)
        outputs = outputs[:, inputs['input_ids'].shape[1]:]
        response = tokenizer.decode(outputs[0])
        response = response.split("</s>")[0]

    tmp_align = {
        "id": i['id'],
        "pure_input": pure_query,
        "image_path": image_path,
        "input": query,
        "output": [
            response
        ],
        "reference": "N/A"
    }
    
    tmp = {
        "question_id": i['id'],
        "prompt": pure_query,
        "text": response,
        "image": image_path
    }

    align_data.append(tmp_align)

    with open(os.path.join(inference_gen_dir, f'{out_file_name}.jsonl'), 'a') as fout:
        fout.write(json.dumps(tmp)+'\n')

with open(os.path.join(align_tds_dir, 'data/', f'{out_file_name}.json'), 'w') as falign:
    json.dump(align_data, falign, indent=4)