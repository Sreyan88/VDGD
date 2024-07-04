import torch
from PIL import Image
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
import sys
import json
from tqdm.auto import tqdm

ckpt_path = "internlm/internlm-xcomposer2-vl-7b"
tokenizer = AutoTokenizer.from_pretrained(ckpt_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(ckpt_path, torch_dtype=torch.bfloat16, trust_remote_code=True).cuda()

model = model.eval()
file_name = sys.argv[1]
out_file_name = sys.argv[2]
sampling = int(sys.argv[3])

current_dir = os.path.dirname(os.path.abspath(__file__))
datasets_dir = os.path.abspath(os.path.join(current_dir, '..', 'datasets'))
align_tds_dir = os.path.abspath(os.path.join(current_dir, '..', 'AlignTDS'))
inference_gen_dir = os.path.abspath(os.path.join(current_dir, '..', 'inference_generations'))

file = open(os.path.join(datasets_dir, f'{file_name}.jsonl'), 'r')
file = file.readlines()
file = [eval(i) for i in file]

align_data = []

for idx, i in tqdm(enumerate(file)):
    images = []

    image_path = os.path.join(datasets_dir, i['image'])
    image = Image.open(image_path).convert('RGB')
    image = model.vis_processor(image)
    images.append(image)
    image = torch.stack(images)

    queries = i['conversations']
    query = ""
    for k in queries:
        if k['from'] == 'human':
            query += k['value'].replace("<image>", "<ImageHere>")
            if "<ImageHere>" not in query:
                query = "<ImageHere> " + query

    if sampling == 0:
        with torch.cuda.amp.autocast():
            response, _ = model.chat(tokenizer, query=query, image=image, max_new_tokens=512, history=[], do_sample=False)

        tmp = {
            "question_id": i['id'],
            "prompt": query.replace("<ImageHere>", ""),
            "text": response,
            "image": image_path
        }

        tmp_align = {
            "id": i['id'],
            "pure_input": query,
            "image_path": image_path,
            "input": query,
            "output": [
                response
            ],
            "reference": "N/A"
        }

        align_data.append(tmp_align)

        with open(os.path.join(inference_gen_dir, f'{out_file_name}.jsonl'), 'a') as fout:
            fout.write(json.dumps(tmp)+'\n')
    else:
        with torch.cuda.amp.autocast():
            response, _ = model.chat(tokenizer, query=query, image=image, max_new_tokens=512, history=[], do_sample=True, temperature=0.9, top_p=0.1)

        tmp = {
            "question_id": i['id'],
            "prompt": query.replace("<ImageHere>", ""),
            "text": response,
            "image": image_path
        }

        tmp_align = {
            "id": i['id'],
            "pure_input": query,
            "image_path": image_path,
            "input": query,
            "output": [
                response
            ],
            "reference": "N/A"
        }

        align_data.append(tmp_align)

        with open(os.path.join(inference_gen_dir, f'{out_file_name}_sd.jsonl'), 'a') as fout:
            fout.write(json.dumps(tmp)+'\n')
    
if sampling == 0:
    with open(os.path.join(align_tds_dir, 'data/', f'{out_file_name}.json'), 'w') as falign:
        json.dump(align_data, falign, indent=4)
else:
    with open(os.path.join(align_tds_dir, 'data/', f'{out_file_name}_sd.json'), 'w') as falign:
        json.dump(align_data, falign, indent=4)