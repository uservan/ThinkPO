import __init__
from utils.data_utils import save_all_results, read_saved_results, load_dataset
from utils.utils import cache_dir, set_global
from utils.load_model import *
from utils.data_utils import *
from utils.eval.eval_utils import check_math_correctness
import random
from tqdm import tqdm

SYSTEM_PROMPT = "Your role as an assistant involves thoroughly exploring questions through a systematic long thinking process before providing the final precise and accurate solutions. This requires engaging in a comprehensive cycle of analysis, summarizing, exploration, reassessment, reflection, backtracing, and iteration to develop well-considered thinking process. Please structure your response into two main sections: Thought and Solution. In the Thought section, detail your reasoning process using the specified format: <|begin_of_thought|> {thought with steps separated with '\n\n'} <|end_of_thought|> Each step should include detailed considerations such as analisying questions, summarizing relevant findings, brainstorming new ideas, verifying the accuracy of the current steps, refining any errors, and revisiting previous steps. In the Solution section, based on various attempts, explorations, and reflections from the Thought section, systematically present the final solution that you deem correct. The solution should remain a logical, accurate, concise expression style and detail necessary step needed to reach the conclusion, formatted as follows: <|begin_of_solution|> {final formatted, precise, and clear solution} <|end_of_solution|> Now, try to solve the following question through the above guidelines:"

save_dirs = set_global('./data/final/NuminaMath-SFT.jsonl')
save_data = read_saved_results(save_dirs)

dataset = load_data('AI-MO/NuminaMath-CoT', 'huggingface')['train']
dataset = dataset.select(random.sample(range(len(dataset)), 16710))
for d in tqdm(dataset):
    question = d['problem']
    response = d['solution'].strip()
    save_data.append({
        'system':SYSTEM_PROMPT, 
        "conversations":[
            {"from": "user", "value": 'Return your final response within \\boxed{}. '+question},
            {"from": "assistant", "value": response}
        ]
        }) 
    save_results(save_dirs, save_data[-1])

