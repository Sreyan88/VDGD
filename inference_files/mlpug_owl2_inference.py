import torch
from PIL import Image
from transformers import TextStreamer
import json
from mplug_owl2.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from mplug_owl2.conversation import conv_templates, SeparatorStyle
from mplug_owl2.model.builder import load_pretrained_model
from mplug_owl2.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
from tqdm.auto import tqdm
import sys

file_name = sys.argv[1]
sampling = int(sys.argv[2])
out_file_name = sys.argv[3]

model_path = 'MAGAer13/mplug-owl2-llama2-7b'
results = []
file = open(f'../datasets/{file_name}.jsonl','r')
file = file.readlines()
file = [eval(i) for i in file]
model_name = get_model_name_from_path(model_path)
tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, None, model_name, load_8bit=False, load_4bit=False, device="cuda")

for i in tqdm(file, total=len(file), desc="Generations"):
    image_file = i['image']
    queries = i['conversations']
    query = ""
    for k in queries:
        if k['from'] == 'human':
            query += k['value']
    conv = conv_templates["mplug_owl2"].copy()
    roles = conv.roles

    image = Image.open(image_file).convert('RGB')
    max_edge = max(image.size) # We recommand you to resize to squared image for BEST performance.
    image = image.resize((max_edge, max_edge))

    image_tensor = process_images([image], image_processor)
    image_tensor = image_tensor.to(model.device, dtype=torch.float16)

    inp = DEFAULT_IMAGE_TOKEN + query

    conv.append_message(conv.roles[0], inp)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt().replace("\n", "").replace("<image>", "")

    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(model.device)
    stop_str = conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    if sampling == 0:
        temperature = 0
        do_sample=False
        top_p = None
    else:
        temperature = 0.9
        do_sample=True
        top_p = 0.1
        
    num_beams=1
    max_new_tokens = 512

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=image_tensor,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            num_beams=num_beams,
            max_new_tokens=max_new_tokens,
            streamer=None,
            use_cache=True,
            stopping_criteria=[stopping_criteria])

    outputs = tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()
    tmp = {
        "question_id": i['id'],
        "prompt": query,
        "text": outputs,
        "image": i['image']
    }

    if sampling == 0:
        with open(f'../inference_generations/{out_file_name}.jsonl','a') as res:
            res.write(json.dumps(tmp) + '\n')
    else:
        with open(f'../inference_generations/{out_file_name}_sd.jsonl','a') as res:
            res.write(json.dumps(tmp) + '\n')