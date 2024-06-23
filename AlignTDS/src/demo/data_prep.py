import pickle
import math
from tqdm import tqdm
import json  
import sys 

pair_name = sys.argv[1] 
i2i_path = sys.argv[2]
i2b_path = sys.argv[3]

 
# pair_name = "llama2_tp"
# i2i_path = "saved_logits/just_eval_1000/llama2/llama2-i2i.pkl"
# i2b_path = "saved_logits/just_eval_1000/llama2_tp/llama2-i2b.pkl"

with open(i2b_path, "rb") as f:
    i2b_data = pickle.load(f)
    # print(i2b_data)

with open(i2i_path, "rb") as f:
    i2i_data = pickle.load(f)

# with open(base_output_json, "r") as f:
#     base_output_data = json.load(f) 

def norm_prob(token_list):
    max_logit = max(token["logit"] for token in token_list)
    sum_exp_logit = sum(math.exp(token["logit"] - max_logit) for token in token_list)

    for token in token_list:
        logit_diff = token["logit"] - max_logit
        token["norm_prob"] = math.exp(logit_diff) / sum_exp_logit

    return token_list
 

all_data = []
for eid in tqdm(range(len(i2b_data))):
    item = {}
    i2b_res = i2b_data[eid]["results"]
    i2i_res = i2i_data[eid]["results"] 
    # print("id:", i2b_data[eid]["id"])
    # print("prompt:", i2b_data[eid]["prompt"])
    # print("probe_text [from falcon-7b-instruct]:", i2b_data[eid]["probe_text"])
    item["id"] = eid
    item["prompt"] = i2b_data[eid]["prompt"]
    item["probe_text"] = i2b_data[eid]["probe_text"]
    item["pure_text"] = i2b_data[eid]["pure_text"]
    formatted_base = {}
    formatted_inst = {}
    print(i2b_res[0])
    for i in range(min(len(i2b_res), len(i2i_res))):
        token_list_1, token_list_2 = i2b_res[i]["tokens"], i2i_res[i]["tokens"]
        token_list_1 = norm_prob(token_list_1)
        token_list_2 = norm_prob(token_list_2)  
        list1 = token_list_1[:100]
        list2 = token_list_2[:100]
        formatted_base[f"position_{i}"] = {"selected_token": list1[0]["token"].replace("\n", "\\n"), "candidates": list1, "prefix": i2b_res[i]["prefix"]}
        formatted_inst[f"position_{i}"] = {"selected_token": list2[0]["token"].replace("\n", "\\n"), "candidates": list2, "prefix": i2i_res[i]["prefix"]}
    # print("concat_text_1:", " ".join(concat_text_1).replace("  ", " "))
    # print("concat_text_2:", " ".join(concat_text_2).replace("  ", " "))
    item["formatted_base"] = formatted_base
    item["formatted_inst"] = formatted_inst
    # item["base_output"] = base_output_data[eid]["output"][0]
    item["instruct_output"] = i2i_data[eid]["final_generated_text"]
    all_data.append(item)

with open(f"src/demo/just_eval+{pair_name}.pkl", "wb") as file:
    print("Saving:", file.name)
    pickle.dump(all_data, file)
    