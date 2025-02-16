import __init__
from utils.data_utils import save_all_results, read_saved_results, load_dataset
from utils.utils import cache_dir, set_global
from utils.data_utils import *
from utils.load_model import *
from utils.model_utils import load_model
from utils.eval.eval_utils import check_math_correctness
import random
from tqdm import tqdm
import re

def qwq_generate(model, tokenizer, max_length ,prompt="How many r in strawberry."):
    messages = [
        {"role": "system", "content": "You are a helpful and harmless assistant. You are Qwen developed by Alibaba. You should think step-by-step."},
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=max_length
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response