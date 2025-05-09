def load_few_shot_cot_prompt():
    with open('prompts/few_shot_cot_prompt.txt', 'r') as file:
        return file.read()

few_shot_cot_prompt = load_few_shot_cot_prompt()



def load_eval_prompt_template():
    with open('prompts/eval_prompt_template.txt', 'r') as file:
        return file.read()

eval_prompt_template = load_eval_prompt_template()


def load_preference_comparison_template():
    with open('prompts/preference_comparison_template.txt', 'r') as file:
        return file.read()

preference_comparison_template = load_preference_comparison_template()

