import __init__
from utils.data_utils import save_all_results, read_saved_results, load_dataset, save_results
from utils.utils import cache_dir, set_global
from utils.load_model import *
from utils.data_utils import *
from utils.eval.eval_utils import check_math_correctness
import random
from tqdm import tqdm
import copy


def generate_is_ok(predict, refrence):
    if not check_math_correctness(refrence, predict): return False
    return True

save_file = set_global(f'./data/final/Bespoke_dpo.jsonl')
save_data = read_saved_results(save_file)
ref_data = load_data('bespokelabs/Bespoke-Stratos-17k', 'huggingface')['train']
MAX_LENGTH = 1024 * 1
tokenizer = load_tokenizer("NovaSky-AI/Sky-T1-32B-Preview")
llm = init_vllm_model("Qwen/Qwen2.5-Math-7B-Instruct", 2)
qwwn_tokenizer = load_tokenizer("Qwen/Qwen2.5-Math-7B-Instruct")
for data in ref_data.select(range(len(save_data), len(ref_data))):
    chosen = data['conversations'][1]['value']
    messages = [
        {"role": "system", "content": "Please reason step by step."},
        {"role": "user", "content":  data['conversations'][0]['value']}
    ]
    prompt = qwwn_tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    # prompt = "You are a helpful and harmless assistant. " + data['conversations'][0]['value']
    rejected = rejected = vllm_generate(llm, prompt, MAX_LENGTH)[0] # ''
    # while len(rejected) == 0 or not generate_is_ok(rejected, chosen):
    #     rejected = vllm_generate(llm, prompt, MAX_LENGTH)[0]
    # messages = [
    #     {"role": "system", "content": data['system']},
    #      {"role": "user", "content":  data['conversations'][0]['value']}
    #     ]
    # prompt = tokenizer.apply_chat_template(
    #         messages,
    #         tokenize=False,
    #         add_generation_prompt=True
    #     )
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




llm = init_vllm_model(model_name, 2)

for data in tqdm(all_data):
    prompt = data['prompt']
    chosen = data['chosen']
    rejected = data['rejected']
    if generate_is_ok(chosen, rejected): 
        save_data.append(data) 

for data in tqdm(ref_data[len(save_data):]):
    prompt = data['prompt']
    chosen = data['chosen']
    rejected = data['rejected']
    for i in range(5):
        reject_ans = vllm_generate(llm, [add_template(model_name, prompt)], MAX_LENGTH)[0]
        if generate_is_ok(reject_ans, chosen): 
            for char in ['<|end_of_solution|>', '<|end_of_thought|>', '<|begin_of_thought|>']:
                reject_ans = reject_ans.replace(char, "").lstrip()
            reject_ans = reject_ans.replace( '<|begin_of_solution|>', "**Final Answer**")
            save_data.append({
                'prompt': prompt,
                'chosen': chosen,
                'rejected': reject_ans
            })
            save_results(save_file, save_data[-1])
            break

