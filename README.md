# VDGD

## Infering using LVLMs
```
cd inference_files/

LLaVA -
python llava_inference.py <model_path> <dataset_name> <output_file_name> <sampling_flag>

python llava_v1_inference.py <dataset_name> <output_file_name> <sampling_flag>

CogVLM -
python cogvlm_inference.py <dataset_name> <output_file_name>

MplugOwl2 -
python mlpug_owl2_inference.py <dataset_name> <output_file_name> <sampling_flag>

InternLM -
python internlm_inference.py <dataset_name> <output_file_name> <sampling_flag>

Supported Arugments:
model_path - this argument is only for llava 1.5 or 1.6 inference file.
dataset_name - file prefix in the `datasets/` folder.
output_file_name - output file name which will be saved at `inference_generations`.
sampling_flag - A 1 or 0 value which will set sampling arguments for inference.
```

## GPT evaluation of LVLM inference
```
cd gpt_evaluations/
python evaluate_gpt.py <inference_file_name>

Supported Arugments:
inference_file_name - file prefix of LVLM output in `inference_generations`.
```

## Logit Analysis of LVLMs
```
cd AlignTDS/ 
sh run.sh <llm_model_name> <shard_size> <num_gpus> <model_generated_dataset_name> <dataset_length>

Example - sh run.sh llava_1.6 126 8 llava_1.6_amber 1004

Supported arguments:
1. llm_model_name - llava_v1, llava_1.5, llava_1.6, cogvlm.

2. shard_size - dataset_length/num_gpus.

3. model_generated_dataset_name - this argument is the name of the file to run logit analysis for in ./AlignTDS/data/.
```

## Categorizing Visual Hallucinations

<<<<<<< Updated upstream
## VDGD Inference
=======

## VDGD Inference
>>>>>>> Stashed changes
