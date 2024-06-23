import json 
from models import DecoderOnlyModelManager
import torch
import torch.nn.functional as F
import argparse
import os 
from tqdm import tqdm 
import pickle 
from transformers import AutoModelForCausalLM, LlamaTokenizer
from PIL import Image
from io import BytesIO

model_pairs = {
    "cogvlm": { "i2i": "THUDM/cogvlm-chat-hf", "i2b": "lmsys/vicuna-7b-v1.5"}
}

llm_templates = {
    "cogvlm": "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: {in_text} ASSISTANT:",
}

def parse_args():
    parser = argparse.ArgumentParser()  
    parser.add_argument('--pair', default="llama2", type=str)
    parser.add_argument('--mode', default="i2i", type=str) # i2i or i2b
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

def load_image(image_file):
    if image_file.startswith("http") or image_file.startswith("https"):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")
    max_edge = max(image.size)
    image = image.resize((max_edge, max_edge))
    return image

def load_images(image_files):
    out = []
    for image_file in image_files:
        image = load_image(image_file)
        out.append(image)
    return out

def get_cogvlm_logits(prompt, image_path, generated_text, pure_text, model, tokenizer, device, top_k, mode="i2i"): 
    generated_tokens = tokenizer.encode(generated_text, return_tensors="pt", add_special_tokens=False).to(device)
    prompt_ids = tokenizer.encode(prompt, return_tensors="pt", add_special_tokens=False).to(device)  

    image = Image.open(image_path).convert('RGB')
    input_by_model = model.build_conversation_input_ids(tokenizer, query=prompt, history=None, images=[image])
    inputs = {
        'input_ids': input_by_model['input_ids'].unsqueeze(0).to(device),
        'token_type_ids': input_by_model['token_type_ids'].unsqueeze(0).to(device),
        'attention_mask': input_by_model['attention_mask'].unsqueeze(0).to(device),
        'images': [[input_by_model['images'][0].to(device).to(torch.bfloat16)]] if image is not None else None,
    }
    if 'cross_images' in input_by_model and input_by_model['cross_images']:
        inputs['cross_images'] = [[input_by_model['cross_images'][0].to(device)]]

    results = []
    input_ids = input_by_model['input_ids'].unsqueeze(0).to(device)
    final_generated_tokens = []

    for i in range(len(generated_tokens[0])):
        with torch.inference_mode():
            if inputs["token_type_ids"].shape != input_ids.shape:
                inputs["token_type_ids"] = torch.cat((inputs["token_type_ids"], torch.zeros(1, 1).to(device)), dim=1)
                model_outputs = model(
                    input_ids=input_ids,
                    token_type_ids=inputs["token_type_ids"],
                    attention_mask=torch.ones(input_ids.shape).to(device),
                    images=inputs["images"]
                )
            else:
                model_outputs = model(
                    input_ids=input_ids,
                    token_type_ids=inputs["token_type_ids"],
                    attention_mask=torch.ones(input_ids.shape).to(device),
                    images=inputs["images"]
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

        prefix = tokenizer.decode(input_ids[0]) # TODO: check results
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
        model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True).to("cuda")
        tokenizer = LlamaTokenizer.from_pretrained("lmsys/vicuna-7b-v1.5", trust_remote_code=True)
        device = model.device
     
    if args.mode == "i2b":
        mm = DecoderOnlyModelManager(model_path, model_name, cache_dir, lora=args.lora, adapt_ckpt=args.adapt_ckpt, is_mplug=(args.pair == "mplug"))
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
            r = get_cogvlm_logits(prompt, image_path, generated_text, pure_text, model, tokenizer, device, args.top_k, mode=args.mode)

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