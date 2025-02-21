import __init__
from utils.utils import *
from utils.data_utils import read_saved_results, save_results
from utils.load_model import *
from utils.data_utils import *
from utils.eval.eval_utils import check_math_correctness
from tqdm import tqdm
import copy


def generate_is_ok(predict, refrence):
    if not check_math_correctness(refrence, predict): return False
    return True

MAX_LENGTH = 1024 * 1

save_file = set_global(f'./data/final/Bespoke_dpo.jsonl')
save_data = read_saved_results(save_file)
ref_data = load_data('bespokelabs/Bespoke-Stratos-17k', 'huggingface')['train']


tokenizer = load_tokenizer("NovaSky-AI/Sky-T1-32B-Preview")
llm = init_vllm_model("Qwen/Qwen2.5-Math-7B-Instruct", 1)
qwwn_tokenizer = load_tokenizer("Qwen/Qwen2.5-Math-7B-Instruct")

for data in tqdm(ref_data):
    chosen = data['conversations'][1]['value']
    messages = [
        {"role": "system", "content": "You are a helpful and harmless assistant. Please reason step by step."},
        {"role": "user", "content":  data['conversations'][0]['value']}
    ]
    prompt = qwwn_tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    rejected = vllm_generate(llm, prompt, MAX_LENGTH)[0] 
    if len(rejected) == 0 or not generate_is_ok(rejected, chosen):
        continue
    rejected_ans = copy.deepcopy(data['conversations'])
    rejected_ans[1]['value'] = rejected
    save_data.append({
        'system': data['system'],
        'chosen': data['conversations'],
        'rejected': rejected_ans
    })
    save_results(save_file, save_data[-1])
    


save_file = set_global(f'./data/final/Bespoke_dpo.jsonl')
save_data = []
all_data = read_saved_results(save_file)
