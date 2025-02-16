# train_dpo.py
import __init__
from datasets import load_dataset
from trl import DPOConfig, DPOTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils.load_model import *

def init_train_model(model_name=''):
    model, tokenizer = init_model(model_name, eval=False)
    model.enable_input_require_grads()
    if tokenizer.pad_token is None:
        token = tokenizer.convert_ids_to_tokens(2)
        print(f"Token for ID 2: {token}")
        tokenizer.pad_token_id = 2 
    return model, tokenizer
model, tokenizer = init_train_model('deepseek-ai/DeepSeek-R1-Distill-Llama-8B')
# model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2-0.5B-Instruct")
# tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B-Instruct")
# model.enable_input_require_grads()
train_dataset = load_dataset("trl-lib/ultrafeedback_binarized", split="train")

training_args = DPOConfig(output_dir="Qwen2-0.5B-DPO", logging_steps=10,
                          per_device_train_batch_size=1, gradient_accumulation_steps=32, bf16=True)
trainer = DPOTrainer(model=model, args=training_args, processing_class=tokenizer, train_dataset=train_dataset)
trainer.train()