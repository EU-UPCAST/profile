from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
device = "auto"

model_name = "jakiAJK/DeepSeek-R1-Distill-Llama-8B_GPTQ-int4"
#model_name = "kaitchup/DeepSeek-R1-Distill-Llama-8B-AutoRound-GPTQ-4bit"
#model_name = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map= device, trust_remote_code= True, torch_dtype= dtype)

model.eval()

chat = [
    { "role": "user", "content": "List any 15 country capitals. Answer in a python list of tuples of form (country, capital)." },
]
chat = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)

input_tokens = tokenizer(chat, return_tensors="pt").to('cuda')

output = model.generate(**input_tokens, 
                        max_new_tokens=100)

output = tokenizer.batch_decode(output)

print(output)
