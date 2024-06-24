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
sampling = int(sys.argv[2])
out_file_name = sys.argv[3]

file = open(f'../datasets/{file_name}.jsonl', 'r')
file = file.readlines()
file = [eval(i) for i in file]

for idx, i in tqdm(enumerate(file)):
    images = []

    image_path = i['image']
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

        print(response)
        print()

        tmp = {
            "question_id": i['id'],
            "prompt": query.replace("<ImageHere>", ""),
            "text": response,
            "image": image_path
        }
        with open(f"../inference_genarations/{out_file_name}.jsonl", "w") as fout:
            fout.write(json.dumps(tmp)+'\n')
    else:
        with torch.cuda.amp.autocast():
            response, _ = model.chat(tokenizer, query=query, image=image, max_new_tokens=512, history=[], do_sample=True, temperature=0.9, top_p=0.1)

        print(response)
        print()

        tmp = {
            "question_id": i['id'],
            "prompt": query.replace("<ImageHere>", ""),
            "text": response,
            "image": image_path
        }
        with open(f"../inference_genarations/{out_file_name}.jsonl", "w") as fout:
            fout.write(json.dumps(tmp)+'\n')