from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from transformers import LlamaTokenizer, LlamaForCausalLM 
import transformers
import torch 
import os
import json  
from transformers import BitsAndBytesConfig
from transformers import StoppingCriteria, StoppingCriteriaList, LogitsProcessor, LogitsProcessorList
from typing import TYPE_CHECKING, Any, Dict, List
from transformers.utils import cached_file
from transformers.trainer import WEIGHTS_NAME, SAFE_WEIGHTS_NAME
from trl import AutoModelForCausalLMWithValueHead
from adapter import init_adapter

# import openai
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)  # for exponential backoff


class EndOfFunctionCriteria(StoppingCriteria):
    """Custom `StoppingCriteria` which checks if all generated functions in the batch are completed."""

    def __init__(self, start_length, eof_strings, tokenizer):
        self.start_length = start_length
        self.eof_strings = eof_strings
        self.tokenizer = tokenizer

    def __call__(self, input_ids, scores, **kwargs):
        """Returns true if all generated sequences contain any of the end-of-function strings."""
        decoded_generations = self.tokenizer.batch_decode(
            input_ids[:, self.start_length :]
        )
        done = []
        for decoded_generation in decoded_generations:
            done.append(
                any(
                    [
                        stop_string in decoded_generation
                        for stop_string in self.eof_strings
                    ]
                )
            )
        return all(done) # Stop when ALL sequences hit the stopping critera
        # return True if True in done # Stop when ANY sequence hits the stopping critera
    
# class BiasWordsLogitsProcessor(LogitsProcessor):
#     def __init__(self, tokenizer, bias_words, bias_value):
#         self.tokenizer = tokenizer
#         self.bias_words = bias_words
#         self.bias_value = bias_value

#     def __call__(self, input_ids, scores):
#         word_indices = []
#         for word in self.bias_words:
#             word_indices += self.tokenizer.encode(word)
#         for idx in word_indices:
#             scores[:, idx] += self.bias_value
#         return scores
 
class BiasWordsLogitsProcessor(LogitsProcessor):
    def __init__(self, tokenizer, bias_words, bias_value, window=3):
        self.tokenizer = tokenizer
        self.bias_word_tokens = [self.tokenizer.encode(word) for word in bias_words]
        self.bias_value = bias_value
        self.window = window

    def __call__(self, input_ids, scores):
        last_k_tokens = input_ids[0, -self.window:].tolist()
        for token_ids in self.bias_word_tokens:
            # Check if the target word tokens are in the last K tokens
            if not any(token_id in last_k_tokens for token_id in token_ids):
                for idx in token_ids:
                    scores[:, idx] += self.bias_value
        return scores
 

class ModelManager:
    def __init__(self, model_path, model_name):
        self.model_path = model_path
        self.model_name = model_name
    
    def load_model(self):
        # Load model from disk
        pass
    
    def infer_logits(self, input_data):
        # Run model inference to get logits
        pass
    
    def infer_generate(self, input_data):
        # Run model inference to generate output
        pass





class DecoderOnlyModelManager(ModelManager):
    
    def __init__(self, model_path, model_name, cache_dir=None, bf16=False, int8=False, bnb4=False, gptq=False, adapt_mode=None, adapt_ckpt=None, lora=0, is_mplug=False):
        super().__init__(model_path, model_name)
        self.cache_dir = cache_dir
        self.bf16 = bf16
        self.bnb4 = bnb4
        self.int8 = int8
        self.gptq = gptq
        self.adapt_mode = adapt_mode
        self.adapt_ckpt = adapt_ckpt
        self.lora = lora
        self.is_mplug = is_mplug
        # if load_type == "local":
        #     assert os.path.exists(model_path)
        #     assert os.path.exists(model_path)+"/"+model_name
        #     assert os.path.exists(model_path)+"/tokenizer/"
        # self.load_type = load_type
    
    
    def _load_adapt_model(self):
        if self.adapt_ckpt == "fixed":
            return 
        print(f"Loading adapt model from {self.adapt_ckpt}...")
        self.adapt_tokenizer =  AutoTokenizer.from_pretrained(self.adapt_ckpt)
        self.adapt_model = AutoModelForSeq2SeqLM.from_pretrained(self.adapt_ckpt)
        if torch.cuda.is_available():
            self.adapt_model = self.adapt_model.to("cuda:0")
        return 

    def load_valuehead_params(self, path_or_repo_id: str) -> Dict[str, torch.Tensor]:
        r"""
        Loads value head parameters from Hugging Face Hub or local disk.

        Returns: dict with keys `v_head.summary.weight` and `v_head.summary.bias`.
        """
        kwargs = {
            "path_or_repo_id": path_or_repo_id,
            "cache_dir": None,
            "token": ""
        }

        try:
            from safetensors import safe_open
            vhead_file = cached_file(filename=SAFE_WEIGHTS_NAME, **kwargs)
            with safe_open(vhead_file, framework="pt", device="cpu") as f:
                return {
                    "v_head.summary.weight": f.get_tensor("v_head.summary.weight"),
                    "v_head.summary.bias": f.get_tensor("v_head.summary.bias")
                }
        except Exception as err:
           print("Failed to load {}: {}".format(SAFE_WEIGHTS_NAME, str(err)))

        try:
            vhead_file = cached_file(filename=WEIGHTS_NAME, **kwargs)
            return torch.load(vhead_file, map_location="cpu")
        except Exception as err:
            print("Failed to load {}: {}".format(WEIGHTS_NAME, str(err)))

        print("Provided path ({}) does not contain valuehead weights.".format(path_or_repo_id))
        return None
         
    def load_model(self, device_str="cuda:0"):
        if self.adapt_mode in ["prefix", "retrieve+prefix"]:
            self._load_adapt_model()
            # self.adapt_model.to(device_str)

        print("loading model: ", self.model_name, "from", self.model_path)
        model_path = self.model_path
        if "@" in self.model_path:
            model_path, revision = model_path.split("@")
        else:
            revision = None         
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, revision=revision, trust_remote_code=True, cache_dir=self.cache_dir, padding_side="left")
        self.special_token_flags = [True, False]

        

        if self.bf16:
            # config = transformers.AutoConfig.from_pretrained(model_path, trust_remote_code=True)
            # config.attn_config['attn_impl'] = 'triton'
            # config.init_device = 'cuda:0' # For fast initialization directly on GPU!
            self.model = AutoModelForCausalLM.from_pretrained(model_path, revision=revision, trust_remote_code=True, device_map="auto", 
                                                            torch_dtype=torch.bfloat16, 
                                                            cache_dir=self.cache_dir)
            # .to(device_str)
        elif self.int8:
            device_map = {
                "transformer.word_embeddings": 0,
                "transformer.word_embeddings_layernorm": 0,
                "lm_head": "cpu",
                "transformer.h": 0,
                "transformer.ln_f": 0,
                "model.layers":"cpu",
                "model.norm":"cpu"
            }
            quantization_config = BitsAndBytesConfig(llm_int8_enable_fp32_cpu_offload=True)
            self.model = AutoModelForCausalLM.from_pretrained(model_path, revision=revision, trust_remote_code=True, 
                                                              device_map=device_map, 
                                                              quantization_config=quantization_config, 
                                                              cache_dir=self.cache_dir)            
        elif self.bnb4:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
            self.model = AutoModelForCausalLM.from_pretrained(model_path, revision=revision, trust_remote_code=True, device_map="auto", 
                                                              quantization_config=bnb_config, cache_dir=self.cache_dir)
        elif self.gptq:
            from auto_gptq import exllama_set_max_input_length
            self.model = AutoModelForCausalLM.from_pretrained(model_path, revision="main", torch_dtype=torch.float16, device_map="auto", trust_remote_code=True, cache_dir=self.cache_dir)            
            if "llama" in model_path.lower():
                self.model = exllama_set_max_input_length(self.model, 4096)
        # elif self.is_mplug:
        #     torch_dtype = torch.float16
        #     from transformers import LlamaForCausalLM, LlamaTokenizer
        #     print("Model mplug load")
        #     self.model = LlamaForCausalLM.from_pretrained(model_path, revision=revision, trust_remote_code=True, device_map="auto", cache_dir=self.cache_dir, torch_dtype=torch_dtype)
        else: 
            torch_dtype = torch.float16
            self.model = AutoModelForCausalLM.from_pretrained(model_path, revision=revision, trust_remote_code=True, device_map="auto", cache_dir=self.cache_dir, torch_dtype=torch_dtype)

        if self.lora == 1:
            self.model = init_adapter(self.model, self.adapt_ckpt, False)
                # self.model = AutoModelForCausalLMWithValueHead.from_pretrained(model_path, revision=revision, trust_remote_code=True, device_map="auto", cache_dir=self.cache_dir, torch_dtype=torch_dtype)
                # vhead_params = self.load_valuehead_params(self.adapt_ckpt)
                # if vhead_params is not None:
                #     print("$$ck successfully loaded vhead_params")
                # else:
                #     print("$$ck vhead_params is empty")
                # self.model.load_state_dict(vhead_params, strict=False)
                # print("Loaded valuehead from checkpoint: {}".format(self.adapt_ckpt))

        print(f"(initial) self.tokenizer.pad_token_id={self.tokenizer.pad_token_id}")
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.tokenizer.padding_side = "left"
        print(f"(updated) self.tokenizer.pad_token_id={self.tokenizer.pad_token_id}")
        self.model.eval()
 
        print("model device:", self.model.device) 
    
    def _adapt_with_prefix(self, input_data, pure_input_data, n=3, args=None): 
        
        if self.adapt_ckpt == "fixed":
            decoded_outputs = [["The answer is: "] for _ in range(len(input_data))]
        # print(input_data_clean)
        # print(len(input_data_clean), input_data)
        else:
            inputs = self.adapt_tokenizer(pure_input_data, return_tensors="pt", add_special_tokens=True, padding=False).to(self.adapt_model.device) 
            outputs = self.adapt_model.generate(
                                input_ids=inputs['input_ids'].to(self.adapt_model.device), 
                                attention_mask=inputs['attention_mask'].to(self.adapt_model.device),
                                # pad_token_id=self.adapt_tokenizer.eos_token_id, 
                                do_sample=False, num_beams=n,
                                # do_sample=True, top_p=0.7, temperature=0.5,
                                num_return_sequences=n,                        
                                max_new_tokens=10, # for the outputs
                            )  
            decoded_outputs = [self.adapt_tokenizer.decode(y, skip_special_tokens=True) for y in outputs]
            decoded_outputs = [decoded_outputs[j:j+n] for j in range(0, len(decoded_outputs), n)]
        input_data_with_prefixes = []
        prefixes = []
        for prompt, outs in zip(input_data, decoded_outputs):
            prefix = outs[0]
            for d in outs:
                if set(d.split()).intersection(set(prompt.split())):
                    prefix = d
                    break 
            input_data_with_prefixes.append(prompt + prefix.strip()) # TODO:
            prefixes.append(prefix.strip())
        return prefixes, input_data_with_prefixes
    
    def infer_generate(self, input_data, args={}, device=None, remarks=None, pure_input_data=None): 
        
        if self.adapt_mode in ["prefix", "retrieve+prefix"]:
            prefixes, input_data = self._adapt_with_prefix(input_data, pure_input_data, args=args)
        
        if not device:
            device = self.model.device
        if type(args) is dict:
            class Args:
                pass  
            args_ = Args()
            args_.__setattr__("num_outputs", args.get("num_outputs", 1))
            args_.__setattr__("beam_size", args.get("beam_size", 1))
            args_.__setattr__("max_output_tokens", args.get("max_output_tokens", 128))
            args_.__setattr__("do_sample", args.get("do_sample", False)) 
            args_.__setattr__("top_p", args.get("top_p", 1.0)) 
            args_.__setattr__("top_k", args.get("top_k", None))
            args_.__setattr__("temperature", args.get("temperature", 1.0)) 
            args_.__setattr__("repetition_penalty", args.get("repetition_penalty", 1.0))
            args_.__setattr__("penalty_alpha", args.get("penalty_alpha", 0))
            args_.__setattr__("no_repeat_ngram_size", args.get("no_repeat_ngram_size", 0))
            args_.__setattr__("length_penalty", args.get("length_penalty", 1.0))
            args_.__setattr__("force_words", args.get("force_words", ""))
            args_.__setattr__("eof_strings", args.get("eof_strings", "---|Query:"))
            args = args_
            
            
        eof_strings = [s.strip() for s in args.eof_strings.split("|")]
        
        # Run Llama model inference to generate output
        if len(input_data) > 1:        
            padding = True 
            # print("Enable padding.")
        else:
            padding = False
        
        n = 1 if args.num_outputs < 0 else args.num_outputs
        if args.num_outputs < 0:
            input_data = [in_text for _ in range(n) for in_text in input_data]
    
        inputs = self.tokenizer(input_data, return_tensors="pt", add_special_tokens=self.special_token_flags[0], padding=padding)
        _, prefix_length = inputs["input_ids"].shape 
        
        # if len(args.force_words.strip()) > 0:
        #     tokenizer_with_prefix_space = AutoTokenizer.from_pretrained(self.model_name, add_prefix_space=True)
        #     bias_words = args.force_words.split("|")
        #     force_words_ids = self.tokenizer(bias_words, add_special_tokens=False).input_ids
        #     # bias_words = ["Yuchen", "Lin", "job", "scientist"]
        #     bias_value = 3  # Modify this value to change the bias intensity
        #     logits_processor = LogitsProcessorList([
        #         BiasWordsLogitsProcessor(tokenizer_with_prefix_space, bias_words, bias_value)
        #     ]) 
        #     # def get_tokens_as_tuple(word):
        #     #     return tuple(tokenizer_with_prefix_space([word], add_special_tokens=False).input_ids[0])
        #     # sequence_bias = {get_tokens_as_tuple("Trump"): 10.0}           
        #     # sequence_bias = {}
        #     # for bw in bias_words:
        #     #     sequence_bias[get_tokens_as_tuple(bw)] = 10.0
        #     sequence_bias = None 
        # else:
        #     force_words_ids = None 
        #     logits_processor = None 
        #     sequence_bias = None
        
        
        
        stopping_criteria = StoppingCriteriaList([EndOfFunctionCriteria(start_length=prefix_length, eof_strings=eof_strings, tokenizer=self.tokenizer)])
        
        
        # print("args.beam_size=", args.beam_size, "n=", n)
        # print('1 if args.do_sample else max(args.beam_size, n)', max(args.beam_size, n))
        # print(inputs['input_ids'][0])
        low_memory = False 
        if args.penalty_alpha > 0:
            # low_memory = True  # if the memory is not enough for you
            pass 
        outputs = self.model.generate(
                        input_ids=inputs['input_ids'].to(device), 
                        attention_mask=inputs['attention_mask'].to(device),
                        pad_token_id=self.tokenizer.pad_token_id, 
                        do_sample=args.do_sample, 
                        top_p=args.top_p, top_k=args.top_k, 
                        temperature=args.temperature,
                        repetition_penalty=args.repetition_penalty, 
                        no_repeat_ngram_size=args.no_repeat_ngram_size, 
                        length_penalty=args.length_penalty, 
                        num_return_sequences=n,
                        num_beams=1 if args.do_sample else max(args.beam_size, n),
                        low_memory =low_memory,
                        # num_beam_groups= 1 if args.do_sample else n,
                        # diversity_penalty= 0.0 if args.do_sample else 10.0,
                        max_new_tokens=args.max_output_tokens, # for the outputs
                        stopping_criteria=stopping_criteria, # Debug
                        # force_words_ids=force_words_ids,
                        # logits_processor=logits_processor,
                        # sequence_bias=sequence_bias,
                        
                    )   
        # decoded_outputs = [self.tokenizer.decode(y).strip() for y in outputs]    
        decoded_outputs = [self.tokenizer.decode(y[prefix_length:], skip_special_tokens=self.special_token_flags[1]) for y in outputs]    
        
        decoded_outputs = [decoded_outputs[j:j+n] for j in range(0, len(decoded_outputs), n)]
        
        if self.adapt_mode in ["prefix", "retrieve+prefix"]:
            decoded_outputs_with_prefixes = []
            for prefix, outputs in zip(prefixes, decoded_outputs):
                tmp_otuputs = [prefix + " " + o for o in outputs]
                decoded_outputs_with_prefixes.append(tmp_otuputs)
                if remarks is not None:
                    remarks.append([prefix])
            decoded_outputs = decoded_outputs_with_prefixes
            
        
        return decoded_outputs



@retry(wait=wait_random_exponential(min=1, max=30), stop=stop_after_attempt(30))
def completion_with_backoff(**kwargs):
    return openai.ChatCompletion.create(**kwargs) 


class OpenAIModelManager(ModelManager):

    def __init__(self, model_name):
        super().__init__(model_name, model_name) 
    
    def load_model(self):
        assert openai.api_key is not None
     
    def infer_generate(self, input_data, args={}, device=None, remarks=[], pure_input_data=None): 
        outputs = [] 
        for input_text in input_data:
            n = args.num_outputs 
            completion = completion_with_backoff(
                model=self.model_name,
                messages=[
                    {"role": "user", "content": input_text}
                ],
                n=n, temperature=0, top_p=1, 
                max_tokens=args.max_output_tokens,
            )
            finished_reason = completion.choices[0].finish_reason
            outputs.append([completion.choices[j].message["content"].strip() for j in range(n)])
            remarks.append({"finished_reason": finished_reason}) 
        return outputs
            