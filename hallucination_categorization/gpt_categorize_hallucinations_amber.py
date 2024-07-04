import pickle
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForSeq2Seq, GenerationConfig
from huggingface_hub.hf_api import HfFolder
from tqdm import tqdm as tqdm
import json
import sys
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

from compute_dist_diff import compute_similarity
import numpy as np

import json
import re
import os

model_generated_dataset_name = sys.argv[1]
object_file_path = sys.argv[2]
gpt_eval_file_name = sys.argv[3]

current_dir = os.path.dirname(os.path.abspath(__file__))
datasets_dir = os.path.abspath(os.path.join(current_dir, '..', 'datasets'))
align_tds_dir = os.path.abspath(os.path.join(current_dir, '..', 'AlignTDS'))
inference_gen_dir = os.path.abspath(os.path.join(current_dir, '..', 'inference_generations'))
gpt_evaluations_dir = os.path.abspath(os.path.join(current_dir, '..', 'gpt_evaluations'))

stop_words = set(stopwords.words('english'))

def search_keywords_in_json(json_file_path, keywords):
    count = 0
    with open(json_file_path, 'r') as file:
        data = json.load(file)
        
        for entry in data:
            conversations = entry.get('conversations', [])
            gpt_values = [convo['value'] for convo in conversations if convo['from'] == 'gpt']

            combined_text = ' '.join(gpt_values)
            # Check if all keywords are found as whole words in the combined text
            found_all_keywords = all(re.search(rf'\b{re.escape(keyword)}\b', combined_text) for keyword in keywords)
            
            if found_all_keywords:
                count = count + 1

    print(f"Total count found : {count}")

def flatten(xss):
    return [x for xs in xss for x in xs]

def check_words_in_string(words, string):
    # Convert all words to lowercase for case-insensitive matching
    words = [word for word in words]
    # Check if all words are in the lowercase version of the string
    return all(word in string for word in words)

def find_matching_entry(data, search_words):
    matching_entries = []
    if len(search_words) != 0:
        for item in data:
            for key, values in item.items():  # Iterate over key-value pairs in the item dictionary
                for value in values:
                    if "value" in value:
                        text_value = value["value"]
                        if value["from"] in ["gpt", "human"] and check_words_in_string(search_words, text_value):
                            matching_entries.append({key: value})
    return matching_entries

def get_word_indices_for_phrase(sentence, start_char_idx, end_char_idx):
    words = sentence.split(" ")
    # Cumulative character lengths will help us map character indices to word indices
    cumulative_lengths = []
    cumulative_length = 0

    for word in words:
        cumulative_length += len(word)
        cumulative_lengths.append(cumulative_length)
        cumulative_length += 1  # Account for the space between words

    # Find the start word index
    start_word_idx = next((i for i, cum_len in enumerate(cumulative_lengths) if cum_len > start_char_idx), None)

    # Adjust start_word_idx because it might point to the next word due to space
    if start_word_idx is not None and cumulative_lengths[start_word_idx] - len(words[start_word_idx]) < start_char_idx:
        start_word_idx -= 1

    # Find the end word index
    end_word_idx = next((i for i, cum_len in enumerate(cumulative_lengths) if cum_len >= end_char_idx), None)

    # Return the range of word indices
    return list(range(start_word_idx, end_word_idx + 1))

with open(os.path.join(align_tds_dir, f"src/demo/just_eval+{model_generated_dataset_name}_tp.pkl"), "rb") as input_file:
    e = pickle.load(input_file)

model_name = "meta-llama/Llama-2-7b-hf"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, access_token="")

top_k_it_images_file_path = "output_text_finetuning_conv_25.json"
top_k_pt_images_file_path = "output_text_pretraining_conv_25.json"

top_k_it_images_dict = json.load(open(top_k_it_images_file_path, "r"))
top_k_pt_images_dict = json.load(open(top_k_pt_images_file_path, "r"))

#Read the questions file
with open(os.path.join(datasets_dir, "amber.jsonl"),"r") as file:
    prompt_list = []
    image_list = []
    for line in file:
        data = json.loads(line)
        image_list.append(data["image"])
        for conv in data["conversations"]:
            if conv["from"] == "human":
                prompt_list.append(conv["value"])

# read evaluation file
with open(os.path.join(gpt_evaluations_dir, f'{gpt_eval_file_name}.json'),"r") as file, open(object_file_path, "r") as obj_file:
    evaluation_list = []
    for line, obj_line in zip(file, obj_file):
        data = json.loads(line)
        obj_data = json.loads(obj_line)
        data["obj_list"] = [obj.strip() for obj in obj_data["predict"].split(",")]
        evaluation_list.append(data)

hallucinations = {}
image_objs = {}
hall_types = {}
img_obj_paths = {}

pre_process_phrase_counter = 0
for item in evaluation_list:
    index = item["image"].split('/')[-1].strip('.jpg').split('_')[-1]
    hall = list(set([i.lstrip().rstrip() for i in item['score']["relation hallucinations"]["tokens"].split(",") if i not in (' ','')] + [i.lstrip().rstrip() for i in item['score']["object hallucinations"]["tokens"].split(",") if i not in (' ','')] + [i.lstrip().rstrip() for i in item['score']["action/verb hallucinations"]["tokens"].split(",") if i not in (' ','')]))
    pre_process_phrase_counter += len(hall)
    image_objs[str(int(index)-1)] = list(set(item["obj_list"]))
    hallucinations[str(int(index)-1)]  = hall
    img_obj_paths[str(int(index)-1)] = item["image"]
    hall_types[str(int(index)-1)] = {
        "object_hallucinations": [i.lstrip().rstrip() for i in item['score']["object hallucinations"]["tokens"].split(",") if i not in (' ','')],
        "relation_hallucinations": [i.lstrip().rstrip() for i in item['score']["relation hallucinations"]["tokens"].split(",") if i not in (' ','')],
        "action_verb_hallucinations": [i.lstrip().rstrip() for i in item['score']["action/verb hallucinations"]["tokens"].split(",") if i not in (' ','')]
    }

# print(f"Pre prcoess phrase counter : {pre_process_phrase_counter}")

import re
pattern_exclude_spaces = r'\w+|[^\w\s]|\n'

mean_absolute_all = []

lang_count = 0
style_count = 0
it_count = 0
vision_count = 0

phrase_counter = 0

zero_count = 0

valid_phrase_counter = 0

failed_idxs_count= 0

for i in tqdm(range(len(e))):
    base = [e[i]['formatted_base']['position_'+str(j)]['candidates'][0]['token_id'] for j in range(len(e[i]['formatted_base']))]
    ft = [e[i]['formatted_inst']['position_'+str(j)]['candidates'][0]['token_id'] for j in range(len(e[i]['formatted_inst']))]

    decoded_full_sentence = tokenizer.decode(ft)
    textlist_ft = re.findall(pattern_exclude_spaces, decoded_full_sentence)
    textlist_ft_joined = " ".join(textlist_ft)
    textlist_ft = list(map(lambda x: x.replace('\n','.'), textlist_ft))
    split_token_map = []
    split_token_map_index = []

    for index, word in enumerate(textlist_ft):
        tokens = tokenizer.tokenize(word)
        split_token_map.extend([word for k in range(len(tokens))])
        split_token_map_index.extend([index for k in range(len(tokens))])

    if str(i) in hallucinations:
        hallucinated_phrases_indices = []

        hall_item = hallucinations[str(i)]

        temp_img_objs = image_objs[str(i)].copy()
        temp_split_img_objs = []
        for obj in temp_img_objs:
            temp_split_img_objs += obj.split(" ")
        temp_split_img_objs = list(set(temp_split_img_objs))
        for phrase in hall_item:
            phrase_counter += 1
            phrase_start = textlist_ft_joined.find(phrase)
            phrase_end = phrase_start + len(phrase)
            word_indices = get_word_indices_for_phrase(textlist_ft_joined, phrase_start, phrase_end) # get the word indices of the hallucinated words
            first_token_indices = [split_token_map_index.index(w_i) for w_i in word_indices]
            hallucinated_phrases_indices = [first_token_indices]

            shifted_marginal_tokens = []
            unshifted_tokens = []
            try:
                for x in flatten(hallucinated_phrases_indices):
                    list1 = e[i]["formatted_base"][f"position_{x}"]
                    list2 = e[i]["formatted_inst"][f"position_{x}"]
                    TR = compute_similarity(list1["candidates"], list2["candidates"], metric="top_rank", top_k=10)
                    if TR > 1:
                        shifted_marginal_tokens.append(x)
                    else:
                        unshifted_tokens.append(x)
            except:
                failed_idxs_count +=1
                continue

            shifted_token_word_list = []
            unshifted_token_word_list = []

            temp_shifted_token_word_list = []
            temp_unshifted_token_word_list = []

            if len(shifted_marginal_tokens) > 0:
                for y in shifted_marginal_tokens:
                    actual_token = e[i]['formatted_inst']['position_'+str(y)]['candidates'][0]['token']
                    actual_token = split_token_map[y]
                    if actual_token.lower() not in stop_words:
                        temp_shifted_token_word_list.append(actual_token)
                        if actual_token in temp_split_img_objs:
                            shifted_token_word_list.append(actual_token)

            if len(unshifted_tokens) > 0:
                for y in unshifted_tokens:
                    actual_token = e[i]['formatted_inst']['position_'+str(y)]['candidates'][0]['token']
                    actual_token = split_token_map[y]
                    if actual_token.lower() not in stop_words:
                        temp_unshifted_token_word_list.append(actual_token)
                        if actual_token in temp_split_img_objs:
                            unshifted_token_word_list.append(actual_token)

            if len(unshifted_token_word_list)>0:
                lang_count+=1

            if len(shifted_token_word_list)>0:
                res1 = (find_matching_entry(top_k_it_images_dict[img_obj_paths[str(i)]], shifted_token_word_list))
                res2 = (find_matching_entry(top_k_pt_images_dict[img_obj_paths[str(i)]], shifted_token_word_list))
                if len(res1) > 0 or len(res2)> 0:
                    it_count += 1
                elif phrase in hall_types[str(i)]["relation_hallucinations"] and phrase not in hall_types[str(i)]["object_hallucinations"]:
                    style_count+=1   
                else:
                    vision_count += 1
            
            if len(unshifted_token_word_list)==0 and len(shifted_token_word_list)==0:
                zero_count+=1
            else:
                valid_phrase_counter+=1

print(f"lang_count: {lang_count} style_count: {style_count} it_count: {it_count} vision_count: {vision_count}")
print(f"Phrase counter: {phrase_counter}")
print(f"Zero counter: {zero_count}")
print(f"Valid phrases: {valid_phrase_counter}")
print(f"failed_idxs_counts: {failed_idxs_count}")