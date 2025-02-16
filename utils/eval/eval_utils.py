from .qwen_math_parser import *


def check_math_correctness(reference, generation):
    if not find_box(generation) or not find_box(reference): return False
    ref = extract_answer(reference)
    answer = strip_answer_string(ref)
    pred = extract_answer(generation)
    pred = strip_answer_string(pred)
    return math_equal(pred, answer)

def check_math_correctness_with_model(reference, generation):
    pass