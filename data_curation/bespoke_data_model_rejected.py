import __init__
from utils.utils import *
from utils.data_utils import save_all_results, read_saved_results, load_dataset, save_results
from utils.load_model import *
from utils.data_utils import *
from utils.eval.eval_utils import check_math_correctness
import random
from tqdm import tqdm
import copy
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

system_prompt = "Your role as an assistant involves thoroughly exploring questions through a systematic long \
thinking process before providing the final precise and accurate solutions. This requires \
engaging in a comprehensive cycle of analysis, summarizing, exploration, reassessment, reflection, \
backtracing, and iteration to develop well-considered thinking process. \
Please structure your response into two main sections: Thought and Solution. \
In the Thought section, detail your reasoning process using the specified format: \
<|begin_of_thought|> {thought with steps separated with '\n\n'} \
<|end_of_thought|> \
Each step should include detailed considerations such as analisying questions, summarizing \
relevant findings, brainstorming new ideas, verifying the accuracy of the current steps, refining \
any errors, and revisiting previous steps. \
In the Solution section, based on various attempts, explorations, and reflections from the Thought \
section, systematically present the final solution that you deem correct. The solution should \
remain a logical, accurate, concise expression style and detail necessary step needed to reach the \
conclusion, formatted as follows: \
<|begin_of_solution|> \
{final formatted, precise, and clear solution} \
<|end_of_solution|> \
Now, try to solve the following question through the above guidelines:" 

model_names = {
    'Bespoke_7b': 'bespokelabs/Bespoke-Stratos-7B', 
    'Bespoke_32b': 'bespokelabs/Bespoke-Stratos-32B', 
    'Instruct-32b':'Qwen/Qwen2.5-32B-Instruct',
    'NuminaMath-32b': set_global(f'./models/sft/NuminaMath_Qwen2.5-3B_sft_lr1e-5/checkpoint-2633') ,
}

ref_data = load_data('bespokelabs/Bespoke-Stratos-17k', 'huggingface')['train']
model_name = 'Bespoke_7b'
gpus = 1

MAX_LENGTH = 1024 * 4
save_file = set_global(f'./data/final/Bespoke_dpo_{model_name}.jsonl')
save_data = read_saved_results(save_file)

tokenizer = load_tokenizer(model_names[model_name])
llm = init_vllm_model(model_names[model_name], gpus)
sampling_params = SamplingParams(
    n=8,             # 生成 8 个回答
    temperature=0.7,  # 控制多样性
    top_p=0.9,       # Nucleus 采样
    max_tokens=MAX_LENGTH,  # 限制回答长度
    use_beam_search=False,
    return_logprobs=True  # 需要返回 logprobs 计算概率
)

for data in ref_data.select(range(len(save_data), len(ref_data))):
    chosen = data['conversations'][1]['value']
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content":  data['conversations'][0]['value']}
    ]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    outputs = llm.generate(messages, sampling_params)

    log_probs_list = []
    for output in outputs[0].outputs:
        tokens = output.token_ids
        log_probs = output.logprobs  # vLLM 直接返回 token 级 log 概率
        
        avg_log_prob = np.mean(log_probs)  # 计算平均 token log 概率
        response_text = output.text.strip()
        log_probs_list.append((avg_log_prob, response_text))

    rejected = max(log_probs_list, key=lambda x: x[0])
    rejected_ans = copy.deepcopy(data['conversations'])
    rejected_ans[1]['value'] = rejected
    save_data.append({
        'system': data['system'],
        'chosen': data['conversations'],
        'rejected': rejected_ans
    })
    save_results(save_file, save_data[-1])
