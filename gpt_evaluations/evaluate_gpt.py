import json
import os
import requests
# from nltk.tokenize import sent_tokenize
import openai
import pandas as pd

import base64
import requests
from tqdm.auto import tqdm
import sys

file_name = sys.argv[1]

current_dir = os.path.dirname(os.path.abspath(__file__))
datasets_dir = os.path.abspath(os.path.join(current_dir, '..', 'datasets'))

def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')


openai.api_key = ''

prompt_abstract_1="Please act as an impartial judge and evaluate the quality of the response provided with respect to the input image. You will rate the quality of the response on multiple aspects, such as Helpfulness, Clarity, Correctness, Depth and Engagement. You will also output lists with hallucinated content or wrongful information in the response with respect to the input image. \n\n##Query: "

prompt_abstract_2="\n\n## Evaluate\n### Aspects  \n- Helpfulness: Rate the response based on how well it addresses the users query about the image and provides a relevant answer. A score of 5 indicates the answer fully aids the user, while a 1 suggests it offers little to no help. \n- Clarity: Rate the response based on how well-structured it is, with ideas presented in a clear and coherent manner. A high score of 5 means the answer is clear and logically structured, while a 1 suggests a disjointed or confusing reply.\n - Correctness: Evaluate the correctness or accuracy of the response provided with respect to the image and the respective question asked about the image. A perfect 5 indicates the response is entirely correct and accurate, while a 1 suggests it has significant errors or has not provided an answer to the question asked at all.\n - Depth: Determine the level of detail and thoroughness in the response. A score of 5 means the answer delves deeply into the aspects of the input image for answering the question, while a 1 indicates it barely scratches the surface. \n\n### Format Given the query and the input image, please rate the quality of the output by scoring it from 1 to 5, individually on **each aspect**. \n- 1: strongly disagree \n- 2: disagree \n- 3: neutral \n- 4: agree \n- 5: strongly agree \n\n You are also asked to output the hallucinated content or wrongful information in the response with respect to the input image with respect to 3 different aspects:\n### Aspects \n- Object Hallucination: Objects in the response that are not present in the image. \n- Action/Verb Hallucination: Actions or verbs in the response that cannot be perceived from the image. For e.g., if the person is just walking and the response contains that the person is jumping. \n- Relation Hallucination: Object-to-object spatial relationships in the response that are not present in the image or cannot be seen or perceived from the image. For example, if a book is on the left of a glass but the response contains it is on the right. \n\nNow, please output your scores, a short rationale, and hallucinated content below in the following json format by filling in the placeholders in []. \n  { 'helpfulness': { 'reason': '[your rationale]', 'score': '[score from 1 to 5]' }, 'clarity': { 'reason': '[your rationale]', 'score': '[score from 1 to 5]' }, 'correctness': { 'reason': '[your rationale]', 'score': '[score from 1 to 5]' }, 'depth': { 'reason': '[your rationale]', 'score': '[score from 1 to 5]' }, 'engagement': { 'reason': '[your rationale]', 'score': '[score from 1 to 5]' }, 'object hallucinations': {'reason': '[elaborate reasoning of hallucination]', 'tokens': '[comma separated list of exact phrases from the response that are hallucinated]'}, 'action/verb hallucinations': {'reason': '[elaborate reasoning of hallucination]', 'tokens': '[comma separated list of exact phrases from the response that are hallucinated]'}, 'relation hallucinations': {'reason': '[elaborate reasoning of hallucination]', 'tokens': '[comma separated list of exact phrases from the response that are hallucinated]'} } \n\nOnly return the json and nothing else."

def prompt_gpt(prompt_input, vision_input, json_output = True):

    response = openai.ChatCompletion.create(model='gpt-4-turbo',
                                                messages=[
                                                            {
                                                            "role": "user",
                                                            "content": [
                                                                {
                                                                "type": "text",
                                                                "text": prompt_input
                                                                },
                                                                {
                                                                "type": "image_url",
                                                                "image_url": {
                                                                    "url": f"data:image/jpeg;base64,{vision_input}"
                                                                }
                                                                }
                                                                       ]
                                                            }
                                                        ],
                                                temperature=0.7,
                                                max_tokens=4096,
                                                response_format={ 'type': 'json_object' }
                                                )
    return response

def get_llm_summary():
    with open(os.path.join(datasets_dir, f'{file_name}.jsonl'),"r") as file:
        prompt_list = []
        image_list = []
        id_list = []
        answer_list = []

        for idx, data in enumerate(file):
            data = json.loads(data)
            image_list.append(os.path.join(datasets_dir, data["image"]))
            prompt_list.append(data["prompt"].replace("\n<image>","").replace("<image>\n",""))
            id_list.append(data["question_id"])
            answer_list.append(data["text"])

    prediction_values = {}

    for i in tqdm(range(len(id_list))):
        prediction_values[id_list[i]] = {
            'Response': answer_list[i],
            'Query': prompt_list[i],
            'Image': image_list[i]    
        }

    instructions = [item["Query"] for item in list(prediction_values.values())]
    responses = [item["Response"] for item in list(prediction_values.values())]
    images = [item["Image"] for item in list(prediction_values.values())]

    dumps_data = {}

    for index, (instruc, pred) in tqdm(enumerate(zip(instructions,responses)), total=len(instructions), unit="Instructions", desc="Prompting GPT-4"):
        try:
            x = prompt_abstract_1 + instruc + "\n\n##Output: " + pred + prompt_abstract_2

            response = prompt_gpt(x, encode_image(images[index]))
            prediction = response['choices'][0]['message']['content'].replace('\n','')

            data = {'instruction': instruc,
                    'answer': pred,
                    'score': eval(prediction.strip('json')),
                    'image': images[index]
                    }
            flag = False
            with open(f'evaluations_{file_name}_correctness.json', 'a') as g:
                g.write(json.dumps(data) + '\n')

        except Exception as e:
            print('Could not be evaluated')
            print(e)
            dumps_data[index] = {
                    'Query': instruc,
                    'Response': pred,
                    'Image': images[index]
                    }

if __name__ == '__main__':
    get_llm_summary()