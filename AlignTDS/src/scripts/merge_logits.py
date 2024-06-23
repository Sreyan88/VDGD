import os
import pickle
import sys 
# path_to_dir = "saved_logits/just_eval_1000/llama2-7b_shards"
# pair_name = "llama"
# split = "i2b"

directory_path = sys.argv[1]
pair_name = sys.argv[2]
split =  sys.argv[3]

# Path to the directory containing the pickle files
# directory_path = f"saved_logits/just_eval_1000//shards"

# Get a list of all pickle files in the directory
pickle_files = [file for file in os.listdir(directory_path+"/shards") if file.endswith('.pkl')]

print(pickle_files)

def extract_start_id(filename):
    s = filename.index("[")+1
    e = filename.index(":")
    start_ind = filename[s:e].strip()
    return int(start_ind)

# Sort the pickle files based on their names
pickle_files.sort(key=lambda x: extract_start_id(x))

print(pickle_files)

# Initialize an empty list to store the merged data
merged_data = []

# Loop through the sorted pickle files and merge their lists
for pickle_file in pickle_files:
    pickle_file_path = os.path.join(directory_path, "shards", pickle_file)
    with open(pickle_file_path, 'rb') as file:
        data_list = pickle.load(file)
        merged_data.extend(data_list)

# Now, the merged_data list contains the merged data from all pickle files
print("Merged data length:", len(merged_data))

# print(merged_data[0].keys())

with open(f"{directory_path}/{pair_name}-{split}.pkl", "wb") as f:
    pickle.dump(merged_data, f)