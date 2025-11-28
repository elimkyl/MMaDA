import torch
from transformers import AutoTokenizer
from models import MMadaModelLM

model = MMadaModelLM.from_pretrained("Gen-Verse/MMaDA-8B-MixCoT", trust_remote_code=True, torch_dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained("Gen-Verse/MMaDA-8B-MixCoT", trust_remote_code=True)