import torch
from prompt_templates import preference_comparison_template, justification_template

# Cross-model majority voting for determining superior reasoning strategies
def cross_model_voting(solutions, eval_models, eval_tokenizers, question, answer, device):
    if len(solutions) < 2:
        return solutions  # Not enough solutions to compare
    
    # Initialize votes for each solution
    votes = [0] * len(solutions)
    
    # Generate all pairs of solutions for comparison
    solution_pairs = []
    for i in range(len(solutions)):
        for j in range(i+1, len(solutions)):
            solution_pairs.append((i, j))
    
    # For each pair, get votes from all models
    for model_idx, (eval_model, eval_tokenizer) in enumerate(zip(eval_models, eval_tokenizers)):
        for i, j in solution_pairs:
            prompt = preference_comparison_template.format(
                question, 
                answer, 
                solutions[i]["text"], 
                solutions[j]["text"]
            )
            
            messages = [
                {"role": "system", "content": "You are a helpful assistant specializing in evaluating mathematical reasoning."},
                {"role": "user", "content": prompt}
            ]
            
            text = eval_tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            model_inputs = eval_tokenizer([text], return_tensors="pt").to(device)
            generated_ids = eval_model.generate(
                **model_inputs,
                max_new_tokens=1024
            )
            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]
            
            response = eval_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            
            # Check preference in the response
            if "Preferred approach: A" in response:
                votes[i] += 1
            elif "Preferred approach: B" in response:
                votes[j] += 1
    
    # Sort solutions by votes (highest first)
    ranked_solutions = [(solution, vote) for solution, vote in zip(solutions, votes)]
    ranked_solutions.sort(key=lambda x: x[1], reverse=True)
    
    return ranked_solutions

# Generate justifications for preferred reasoning strategies
def generate_justifications(ranked_solutions, eval_model, eval_tokenizer, question, device):
    justified_pairs = []
    
    # For each pair where one solution is preferred over another
    for i in range(len(ranked_solutions)-1):
        winner_solution = ranked_solutions[i][0]
        loser_solution = ranked_solutions[i+1][0]
        
        # Generate justification for why the winning solution is better
        prompt = justification_template.format(
            question,
            winner_solution["text"]
        )
        
        messages = [
            {"role": "system", "content": "You are a helpful assistant specializing in explaining mathematical reasoning."},
            {"role": "user", "content": prompt}
        ]
        
        text = eval_tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        model_inputs = eval_tokenizer([text], return_tensors="pt").to(device)
        generated_ids = eval_model.generate(
            **model_inputs,
            max_new_tokens=1024
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        
        justification = eval_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        justified_pairs.append({
            "question": question,
            "chosen": winner_solution["text"],
            "rejected": loser_solution["text"],
            "justification": justification
        })
    
    return justified_pairs