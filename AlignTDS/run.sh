export HF_HOME=..//cache

python -c "from huggingface_hub.hf_api import HfFolder; HfFolder.save_token('')"

module load cuda/11.8.0

if [[ $1 == "llava_1.5" ]]; then
    sh align_logits.sh $1 $2 $3 $4 $5 
elif [[ $1 == "llava_v1" ]]; then
    sh align_logits_llava_v1.sh $1 $2 $3 $4 $5
elif [[ $1 == "llava_1.6" ]]; then
    sh align_logits.sh $1 $2 $3 $4 $5
elif [[ $1 == "cogvlm" ]]; then
    sh align_logits_cogvlm.sh $1 $2 $3 $4 $5
fi