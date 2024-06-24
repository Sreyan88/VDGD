from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path
from llava.eval.run_llava import eval_model_loop, eval_model
import json
import os
import sys

model_path = "liuhaotian/llava-llama-2-7b-chat-lightning-lora-preview"

file_name = sys.argv[1]
sampling = int(sys.argv[2])
out_file_name = sys.argv[3]

with open(f"../datasets/{file_name}.jsonl","r") as file:
    prompt_list = []
    image_list = []

    for line in file:
        data = json.loads(line)
        image_list.append(data["image"])
        for conv in data["conversations"]:
            if conv["from"] == "human":
                prompt_list.append(conv["value"])
                break

    preds = []

    print(prompt_list[-1])

    if sampling == 0:
        args = {
            "model_path": model_path,
            "model_base": "meta-llama/Llama-2-7b-chat-hf",
            "model_name": get_model_name_from_path(model_path),
            "queries": prompt_list,
            "conv_mode": None,
            "image_files": image_list,
            "sep": ",",
            "do_sample": False,
            "temperature": 0,
            "top_p": None,
            "num_beams": 1,
            "max_new_tokens": 512
        }

        preds, final_prompts = eval_model_loop(args)
        with open(f"../inference_generations/{out_file_name}.jsonl", "w") as file, open(f"../AlignTDS/data/{out_file_name}.json", "w") as file_tds:
            json_data = []
            count = 0
            for p, pred, fp, imp in zip(prompt_list, preds, final_prompts, image_list):
                try:
                    data= {
                        "id": count,
                        "pure_input": p.replace("<image>","").strip("\n"),
                        "image_path": imp,
                        "input": fp,
                        "output": [pred],
                        "reference": "N/A"
                    }

                    file.write(json.dumps({"question_id": count,
                                                    "prompt": p.replace("<image>","").strip("\n"),
                                                    "text": pred,
                                                    "model_id": "N/A",
                                                    "image": imp,
                                                    "metadata": {}}) + "\n")
                    file.flush()
                    json_data.append(data)
                    count+=1
                except:
                    print(f"Error at id: {count}")
                    continue
            json.dump(json_data, file_tds, indent=4)
    else:
        args = {
            "model_path": model_path,
            "model_base": "meta-llama/Llama-2-7b-chat-hf",
            "model_name": get_model_name_from_path(model_path),
            "queries": prompt_list,
            "conv_mode": None,
            "image_files": image_list,
            "sep": ",",
            "do_sample": True,
            "temperature": 0.5,
            "top_p": 0.5,
            "num_beams": 1,
            "max_new_tokens": 512
        }

        preds, final_prompts = eval_model_loop(args)

        with open(f"../inference_generations/{out_file_name}.jsonl", "w") as file, open(f"../AlignTDS/data/{out_file_name}.json", "w") as file_tds:
            json_data = []
            count = 0
            for p, pred, fp, imp in zip(prompt_list, preds, final_prompts, image_list):
                try:
                    data= {
                        "id": count,
                        "pure_input": p.replace("<image>","").strip("\n"),
                        "image_path": imp,
                        "input": fp,
                        "output": [pred],
                        "reference": "N/A"
                    }

                    file.write(json.dumps({"question_id": count,
                                                    "prompt": p.replace("<image>","").strip("\n"),
                                                    "text": pred,
                                                    "model_id": "N/A",
                                                    "image": imp,
                                                    "metadata": {}}) + "\n")
                    file.flush()
                    json_data.append(data)
                    count+=1
                except:
                    print(f"Error at id: {count}")
                    continue
            json.dump(json_data, file_tds, indent=4)