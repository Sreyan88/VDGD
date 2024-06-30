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

from transformers import AutoModelForCausalLM, AutoTokenizer

from PIL import Image
import numpy as np
import pickle

from transformers import set_seed
from vcd_utils.vcd_add_noise import add_diffusion_noise
from vcd_utils.vdgd_sample import evolve_vcd_sampling
evolve_vcd_sampling()

llm_templates = {
    "lmsys/vicuna-7b-v1.5": "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: {in_text} ASSISTANT:",
    "meta-llama/Llama-2-7b-chat-hf": "[INST] <<SYS>>\nYou are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.\n<</SYS>>\n\n{in_text} [/INST]"
}

desc_prompt = "I have been given an image to complete the task described as: {inst}.\n\nTo help me complete the task, describe the given image in detail. In case of real-world scenes, please include all foreground and background objects in the description, their properties (like color, shape, etc.), their relations with other objects, their count, and all other components in the image. In case of non-real-world scenes, like charts, graphs, tables, etc., please describe the table, mention all numbers (if any), mention the written text, and all other details."

def eval_model(args):
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    compute_dtype = torch.float16
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)
    amateur_model = AutoModelForCausalLM.from_pretrained(args.amateur_model_path).cuda()
    amateur_tokenizer = AutoTokenizer.from_pretrained(args.amateur_model_path)

    questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "a")
    file_path = f'description_logits/logits_{question_file}.pkl'
    logits_list = []
    for index, line in tqdm(enumerate(questions)):
        idx = line["id"]
        image_file = line["image"]
        # qs = line["text"]
        for conv in line["conversations"]:
            if conv["from"] == "human":
                inst = conv["value"]
                if "<image>" not in inst:
                    inst = "<image>\n" + inst                
                qs = desc_prompt.format(inst=inst)
                am_prompt = llm_templates[args.amateur_model_path].format(in_text=qs.replace("<image>", "").strip())
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
        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
        amateur_input_ids = amateur_tokenizer.encode(am_prompt, return_tensors="pt", add_special_tokens=True).to("cuda")
        image = Image.open(os.path.join(args.image_folder, image_file))
        image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
        
        if args.use_cd:
            image_tensor_cd = add_diffusion_noise(image_tensor, args.noise_step)
        else:
            image_tensor_cd = None      
        
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
        input_token_len = input_ids.shape[1]
        with torch.inference_mode():
            greedy_output = model.generate(
                input_ids,
                images=image_tensor.unsqueeze(0).half().cuda(),
                images_cd=(image_tensor_cd.unsqueeze(0).half().cuda() if image_tensor_cd is not None else None),
                amateur_model=amateur_model,
                amateur_input_ids=amateur_input_ids,
                cd_alpha = args.cd_alpha,
                cd_beta = args.cd_beta,
                use_dd = args.use_dd,
                use_ori = args.use_ori,
                use_dd_unk = args.use_dd_unk,
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature if args.temperature > 0 else 1.0,
                top_p=args.top_p,
                top_k=args.top_k,
                max_new_tokens=1024,
                return_dict_in_generate=True, 
                output_scores=True,
                output_logits=True,
                use_cache=True)
            logits = greedy_output["logits"]
            logits = tuple(tensor.cpu() for tensor in logits)
            logits_list.append(logits)
            output_ids = greedy_output[0]

        n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
        if n_diff_input_output > 0:
            print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
        outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
        outputs = outputs.strip()
        if outputs.endswith(stop_str):
            outputs = outputs[:-len(stop_str)]
        outputs = outputs.strip()

        ans_file.write(json.dumps({"question_id": idx,
                                   "prompt": cur_prompt,
                                   "text": outputs,
                                   "model_id": model_name,
                                   "image": image_file,
                                   "metadata": {}}) + "\n")
        ans_file.flush()
    with open(file_path, 'wb') as f:
        pickle.dump(logits_list, f)

    ans_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="facebook/opt-350m")
    parser.add_argument("--amateur_model_path", type=str, default="lmsys/vicuna-7b-v1.5")
    parser.add_argument("--model_base", type=str, default=None)
    parser.add_argument("--image_folder", type=str, default="")
    parser.add_argument("--question_file", type=str, default="./data/POPE/coco/coco_pope_adversarial.json")
    parser.add_argument("--answers_file", type=str, default="./output/llava15_coco_pope_adversarial_setting.jsonl")
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
    args.temperature = 0.0000000000000000000001
    args.top_p = None
    args.top_k = None
    args.do_sample = True
    default_args = copy.deepcopy(args)

    args.answers_file = answers_file.replace('setting', 'default')
    eval_model(args)