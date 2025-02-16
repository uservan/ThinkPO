project_dir = '/home/wxy320/ondemand/program/o1-reproduce'
cache_dir = '/scratch/pbsjobs/wxy320/huggingface'
hug_token = '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'
SYSTEM_PROMPT = {
    "Qwen/Qwen2-7B-Instruct": "You are a helpful and harmless assistant. You are Qwen developed by Alibaba. You should think step-by-step.",
    "Qwen/QwQ-32B-Preview": "You are a helpful and harmless assistant. You are Qwen developed by Alibaba. You should think step-by-step.",
    "Qwen/Qwen2.5-72B-Instruct": "You are a helpful and harmless assistant. You are Qwen developed by Alibaba. You should think step-by-step.",
    "Qwen/Qwen2.5-32B-Instruct": "You are a helpful and harmless assistant. You are Qwen developed by Alibaba. You should think step-by-step.",
    "Qwen/Qwen2.5-7B-Instruct": "You are a helpful and harmless assistant. You are Qwen developed by Alibaba. You should think step-by-step.",
    "NovaSky-AI/Sky-T1-32B-Preview": "Your role as an assistant involves thoroughly exploring questions through a systematic long \
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
        Now, try to solve the following question through the above guidelines:",
    "openai/o1-mini": "Question: {input}\nAnswer: ",
    "openai/o1-preview": "Question: {input}\nAnswer: ",
    "openai/gpt-4o-mini": "User: {input}\nPlease reason step by step, and put your final answer within \\boxed{{}}.\n\nAssistant:",
}