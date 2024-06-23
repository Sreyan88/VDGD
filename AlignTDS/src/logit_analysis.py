import json 
from models import DecoderOnlyModelManager
import torch
import torch.nn.functional as F
import argparse
import os 
from tqdm import tqdm 
import pickle 
from transformers import AutoProcessor
from PIL import Image
from io import BytesIO

from llava.utils import disable_torch_init
from llava.eval.run_llava import load_image, load_images

from llava.constants import IMAGE_TOKEN_INDEX

from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import (
    process_images,
    tokenizer_image_token,
    get_model_name_from_path,
)

model_pairs = {
    "llava_1.6": { "i2i": "liuhaotian/llava-v1.6-vicuna-7b", "i2b": "lmsys/vicuna-7b-v1.5"},
    "llava_1.5": { "i2i": "liuhaotian/llava-v1.5-7b", "i2b": "meta-llama/Llama-2-7b-chat-hf"},
}

llm_templates = {
    "llava_1.6": "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: {in_text} ASSISTANT:",
    "llava_1.5": "[INST] <<SYS>>\nYou are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.\n<</SYS>>\n\n{in_text} [/INST]",
}

def parse_args():
    parser = argparse.ArgumentParser()  
    parser.add_argument('--pair', default="llama2", type=str)
    parser.add_argument('--mode', default="i2i", type=str)
    parser.add_argument('--top_k', default=30, type=int)
    parser.add_argument('--start', default=1, type=int)
    parser.add_argument('--end', default=5, type=int)
    parser.add_argument('--data_file', default="result_dirs/just_results/Llama-2-7b-chat-hf.json", type=str)
    parser.add_argument('--i2i_pkl_file', type=str)
    parser.add_argument('--logits_folder', default="saved_logits/just_eval/shards/", type=str)
    parser.add_argument('--enable_template', action="store_true")
    parser.add_argument('--lora', required=True, type=int)
    parser.add_argument('--adapt_ckpt', type=str, default=None)
    parser.add_argument('--llm', type=str, required=True)
    return parser.parse_args()

def get_llava_logits(prompt, image_path, generated_text, pure_text, model, tokenizer, image_processor, device, top_k, mode="i2i"): 
    disable_torch_init()
    
    generated_tokens = tokenizer.encode(generated_text, return_tensors="pt", add_special_tokens=False).to(device)
    prompt_ids = tokenizer.encode(prompt, return_tensors="pt", add_special_tokens=False).to(device)
    images = load_images([image_path])
    image_sizes = [x.size for x in images]

    images_tensor = process_images(
        images,
        image_processor,
        model.config
    ).to(model.device, dtype=torch.float16)

    input_ids = (
        tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
        .unsqueeze(0)
        .cuda()
    )    

    results = []
    final_generated_tokens = []

    for i in range(len(generated_tokens[0])):
        with torch.inference_mode():
            model_outputs = model(
                input_ids=input_ids,
                images=images_tensor,
                image_sizes=image_sizes
            )
            logits = model_outputs.logits
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)

            top_k_indices = logits.topk(k=top_k).indices

        result = []
        logits = logits.cpu().numpy()
        probs = probs.cpu().numpy()
        top_k_indices = top_k_indices.cpu().numpy()
        prefix = tokenizer.decode(prompt_ids[0])
        for j in range(top_k):
            d = {}
            d["token_id"] = top_k_indices[0][j]
            d["token"] = tokenizer.decode(d["token_id"])
            d["logit"] = logits[0][d["token_id"]]
            d["prob"] = probs[0][d["token_id"]]
            result.append(d)
        results.append({"tokens":result, "prefix": prefix})
        selected_token = generated_tokens[0][i]
        final_generated_tokens.append(selected_token)
        new_token_tensor = torch.tensor([[selected_token]]).to(device)
        prompt_ids = torch.cat((prompt_ids, new_token_tensor), dim=1).to(device)
        input_ids = torch.cat((input_ids, new_token_tensor), dim=1).to(device)
    final_generated_text = tokenizer.decode(final_generated_tokens, add_special_token=False)
    return results, final_generated_text

def get_logits(prompt, generated_text, model, tokenizer, device, top_k, mode="i2i"):
    generated_tokens = tokenizer.encode(generated_text, return_tensors="pt", add_special_tokens=False).to(device)
    results = []
    input_ids = tokenizer.encode(prompt, return_tensors='pt', add_special_tokens=True).to(device)
    final_generated_tokens = []
    for i in range(len(generated_tokens[0])):
        with torch.no_grad():
            model_outputs = model(input_ids)
            logits = model_outputs.logits
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            top_k_indices = logits.topk(k=top_k).indices
        result = []
        logits = logits.cpu().numpy()
        probs = probs.cpu().numpy()
        top_k_indices = top_k_indices.cpu().numpy()
        prefix = tokenizer.decode(input_ids[0])
        for j in range(top_k):
            d = {}
            d["token_id"] = top_k_indices[0][j]
            d["token"] = tokenizer.decode(d["token_id"])
            d["logit"] = logits[0][d["token_id"]]
            d["prob"] = probs[0][d["token_id"]]
            result.append(d)
        results.append({"tokens":result, "prefix": prefix})
        selected_token = generated_tokens[0][i]
        if mode == "i2i":
            final_generated_tokens.append(selected_token)
        new_token_tensor = torch.tensor([[selected_token]]).to(device)
        input_ids = torch.cat((input_ids, new_token_tensor), dim=1).to(device)

    if mode == "i2i":
        final_generated_text = tokenizer.decode(final_generated_tokens, add_special_token=False)
        return results, final_generated_text
    elif mode == "i2b":
        return results
    else:
        raise NotImplementedError

def main():

    args = parse_args()

    cache_dir = None
    
    print(f"Args.pair : {args.pair}")
    model_path = model_pairs[args.pair][args.mode]
    model_name = "x"
      
    with open(args.data_file) as f:
        instruct_data = json.load(f) 
        
    ids = []
    input_texts = [] 
    pure_texts = []
    output_texts = [] 
    image_paths = []
    for ind in range(len(instruct_data)): 
        item = instruct_data[ind] 
        ids.append(item["id"])
        if args.mode == "i2i":
            pure_texts.append(llm_templates[args.llm].format(in_text=item["pure_input"]))
            input_texts.append(item["input"])
            image_paths.append(item["image_path"])
        elif args.mode == "i2b":
            in_text = item["pure_input"]
            pure_texts.append(in_text)
            if args.enable_template:
                in_text = llm_templates[args.llm].format(in_text=in_text)
            input_texts.append(in_text)
            image_paths.append("")
        else:
            raise NotImplementedError
        
        output_texts.append(item["output"][0])

    logit_results = [] 

    if args.mode == "i2i":
        model_name = get_model_name_from_path(model_path)
        tokenizer, model, image_processor, context_len = load_pretrained_model(
            model_path, None, model_name
        )
        device = model.device
     
    if args.mode == "i2b":
        mm = DecoderOnlyModelManager(model_path, model_name, cache_dir, lora=args.lora, adapt_ckpt=args.adapt_ckpt)
        mm.load_model()
        model = mm.model
        device = model.device
        tokenizer = mm.tokenizer 
        output_texts = []
        assert os.path.exists(args.i2i_pkl_file)
        with open(args.i2i_pkl_file, "rb") as file:
            i2i_results = pickle.load(file) 
        for d in i2i_results:
            output_texts.append(d["final_generated_text"])
    
    
    s = args.start
    e = args.end

    for ind, prompt, generated_text, pure_text, image_path in tqdm(zip(ids[s:e], input_texts[s:e], output_texts[s:e], pure_texts[s:e], image_paths[s:e]), total=e-s, desc=args.mode): 
        
        d = {"id": ind, "prompt": prompt, "probe_text": generated_text, "pure_text": pure_text, "image_path": image_path}

        if args.mode=="i2b":

            r = get_logits(prompt, generated_text, model, tokenizer, device, args.top_k, mode=args.mode)
        else:
            r = get_llava_logits(prompt, image_path, generated_text, pure_text, model, tokenizer, image_processor, device, args.top_k, mode=args.mode)

        if args.mode == "i2i":
            d["results"], d["final_generated_text"] = r
        elif args.mode == "i2b":
            d["results"] = r
        logit_results.append(d) 

    # Save logit_results to i2i.pkl
    with open(os.path.join(args.logits_folder, f"{args.pair}-{args.mode}.[{args.start}:{args.end}].pkl"), "wb") as file:
        pickle.dump(logit_results, file) 

if __name__ == "__main__":
    main()