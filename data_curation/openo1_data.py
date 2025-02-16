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

def contains_chinese(text):
    return any('\u4e00' <= char <= '\u9fff' for char in text)
def len_lower(text):
    global MAX_LENGTH, tokenizer
    inp = tokenizer(text, add_special_tokens=False)
    if len(inp["input_ids"]) > MAX_LENGTH-75:
        return False
    return True


# prompts inspired by https://www.databricks.com/blog/LLM-auto-eval-best-practices-RAG
special_tokens=['<Thought>', '</Thought>','<Output>', '</Output>']
MAX_LENGTH = 1024 * 2 # tokenizer.model_max_length
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B" ,cache_dir=cache_dir,)



def extract_output(text):
    pattern = r'<Output>(.*?)</Output>'
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1)
    else:
        return None
    
def extract_thought(text):
    pattern = r'<Thought>(.*?)</Thought>'
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1)
    else:
        return None

def parse_qa_json(text):
    matches = re.findall(r"<.*?\>", text, re.DOTALL)
    if len(matches) > 0:
        try:
            r = json.loads(matches[-1])
        except:
            return None
        return r
    return None


def transform_answer(model, prompt, response):
    new_reponse = extract_output(response)
    p = f"""
        ### Prompt
        You are a formatting assistant. Your task is to convert answers from the dataset into a specific format. Please follow the instructions below for each input.
        ### Instructions:
        1. You will receive a `prompt` and an `answer` in the input data.
        2. Extract the correct mathematical or numerical result from the `answer` field.
        3. Reformat the answer by placing the correct result inside \\boxed{{}}.
        4. The output should follow this format: <Output>**Final Answer**\n\nThe answer is: $\\boxed{{correct answer}}$.</Output>
        ### Example Input:
        {{
            "prompt": "A line segment begins at \\((2, 5)\\). It is 13 units long and ends at the point \\((10, y)\\) where \\(y > 0\\). What is the value of \\(y\\)?",
            "answer": "\n\\( y = 5 + \\sqrt{{105}} \\) or approximately \\( y \\approx 15.247 \\)\n"
        }}
        changed the answer into this format: 
            <Output>**Final Answer** \n\n\\( y = \\boxed{{5 + \\sqrt{{105}}}} \\) or approximately \\( y \\approx \\boxed{{15.247}} \\)</Output>
        ### Now change the answer of this data
        {{"prompt": "{prompt}", "answer": "{new_reponse}"}}
        The result of the changed data is 
        """
    try:
        o = model.generate(prompt=p)['output']
        # print(o)
        final_reponse = extract_thought(response.split('<Output>')[0]) + extract_output(o)
    except Exception as e:
        print(e)
        final_reponse = '-1'
    return final_reponse

train_dataset = load_data('/users/PDS0352/wyang107/project/LCEG/o1/data/generate/OpenO1-SFT-Pro.jsonl', mode='json')['train']
train_dataset =  train_dataset.filter(lambda example: not contains_chinese(example['prompt']) and len_lower(example['response']))
models=[ load_model('gpt-4o-mini-2024-07-18', temperature=0.1), load_model('gpt-4o-2024-11-20', temperature=0.1)] # gpt-4o-mini-2024-07-18   gpt-4o-2024-11-20
save_path = set_global('./data/generate/OpenO1-SFT-Pro-Filter2.jsonl')
final_reponses = read_saved_results(save_path)
for i, d in tqdm(enumerate(train_dataset), total=len(train_dataset)): 
    if i < len(final_reponses) and final_reponses[i]['response'] != '-1':
        continue
    prompt, response = d['prompt'], d['response']
    for model in models:
        final_reponse = transform_answer(model, prompt, response)
        if final_reponse != '-1': break
    if final_reponse == '-1':
        new_reponse = extract_output(response)
        # p = judge_prompt.format(question=prompt, answer=new_reponse)
        final_reponse = extract_thought(response.split('<Output>')[0]) + '**Final Answer** \n'+ new_reponse  
    if i< len(final_reponses):
        final_reponses[i] = {'prompt':prompt, 'response': final_reponse}
        save_all_results(save_path, final_reponses)
    else:
        final_reponses.append({'prompt':prompt, 'response': final_reponse})
        save_results(save_path, final_reponses[-1])
    if i%1000 == 0: print(f'finish {i}/{len(train_dataset)}')
        
        
    
    
   