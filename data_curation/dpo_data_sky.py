import __init__
from utils.data_utils import save_all_results, read_saved_results, load_dataset, save_results
from utils.utils import cache_dir, set_global
from utils.load_model import *
from utils.eval.eval_utils import check_math_correctness
import random
from tqdm import tqdm

def generate_is_ok(predict, refrence):
    if not check_math_correctness(refrence, predict): return False
    return True

save_file = set_global(f'./data/final/NuminaMath-CoT_O1_Qwq_sky.jsonl')
save_data = []
all_data = read_saved_results(save_file)
ref_data = read_saved_results(set_global(f'./data/final/NuminaMath-CoT_O1_Qwq1.jsonl'))

MAX_LENGTH = 1024 * 4
model_name = "NovaSky-AI/Sky-T1-32B-Preview"
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

