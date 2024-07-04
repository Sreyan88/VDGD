current_directory=$(pwd)

# i2i   
instruct_data_file="$current_directory/data/$4.json"
logits_folder="$current_directory/saved_logits/just_eval_1000/$4/shards/"
# i2i
mkdir -p $logits_folder
n_shards=$3 # or 1 if you only have one gpu
shard_size=$2 # or 1000 if you only have one gpu
start_gpu=0
for ((start = 0, end = (($shard_size)), gpu = $start_gpu; gpu < $n_shards+$start_gpu; start += $shard_size, end += $shard_size, gpu++)); do
    if (( $end > $5 )); then
        end=$5
    fi

    CUDA_VISIBLE_DEVICES=$gpu python src/logit_analysis_llava_v1.py \
                --data_file $instruct_data_file \
                --logits_folder $logits_folder \
                --pair $1 \
                --lora 0 \
                --mode i2i \
                --llm $1\
                --start $start --end $end &
    pids[$gpu]=$!
done

echo "Waiting for pids to exit"

for ((i=0; i<$n_shards; i++)); do
    wait ${pids[$i]}
done

echo "Merging the shards"

python src/scripts/merge_logits.py saved_logits/just_eval_1000/$4/ $4 i2i

logits_folder="saved_logits/just_eval_1000/$4_tp/shards/"
mkdir -p $logits_folder
n_shards=$3
shard_size=$2
start_gpu=0
for ((start = 0, end = (($shard_size)), gpu = $start_gpu; gpu < $n_shards+$start_gpu; start += $shard_size, end += $shard_size, gpu++)); do
    if (( $end > $5 )); then
        end=$5
    fi
    
    CUDA_VISIBLE_DEVICES=$gpu python src/logit_analysis_llava_v1.py \
                --data_file $instruct_data_file \
                --enable_template \
                --logits_folder $logits_folder \
                --pair $1 \
                --lora 0 \
                --mode i2b \
                --llm $1\
                --i2i_pkl_file saved_logits/just_eval_1000/$4/$4-i2i.pkl \
                --start $start --end $end &
    pids[$gpu]=$!
done

echo "Waiting for pids to exit"

for ((i=0; i<$n_shards; i++)); do
    wait ${pids[$i]}
done

echo "Merging the shards"

python $current_directory/src/scripts/merge_logits.py $current_directory/saved_logits/just_eval_1000/$4_tp/ $4 i2b

python $current_directory/src/demo/data_prep.py $4_tp $current_directory/saved_logits/just_eval_1000/$4/$4-i2i.pkl $current_directory/saved_logits/just_eval_1000/$4_tp/$4-i2b.pkl

arg3=$2

arg3=$((arg3 * n_shards))

if (( arg3 > $5 )); then
    arg3=$5
fi

python $current_directory/src/demo/generate_html.py $4_tp $arg3