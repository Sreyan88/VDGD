# VDGD

## Infering using LVLMs

## GPT evaluation of LVLM inference

## Logit Analysis of LVLMs
```
cd AlignTDS/ 
sh run.sh <llm_model_name> <shard_size> <num_gpus> <model_generated_dataset_name> <dataset_length>

Example - sh run.sh llava_1.6 126 8 llava_1.6_amber 1004

cd ../hallucination_category_algorithm/



Supported arguments:
1. llm_model_name - llava_v1, llava_1.5, llava_1.6, cogvlm.

2. shard_size - dataset_length/num_gpus.

3. model_generated_dataset_name - this argument is the name of the file to run logit analysis for in ./AlignTDS/data/.
```

## Categorizing Visual Hallucinations

## VDGD Inference