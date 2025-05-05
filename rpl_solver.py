from model import State, VisionLanguageModel
from mcts import mcts_search
from preference_learning import cross_model_voting, generate_justifications

def solve_math_reasoning_vlm_with_rpl(image_data, text_prompt, model, generation_config, processor, 
                                      eval_models, eval_tokenizers, question, answer, n_iterations, device):
    """Enhanced version of the original solve function that incorporates RPL"""
    image_feat = image_data

    init_state = State(
        image_feat=image_feat,
        text_context=text_prompt,
        solution_steps=[]
    )

    vlm = VisionLanguageModel(model, processor)

    # Run MCTS to generate diverse reasoning paths
    root, steps, solution, diverse_solutions, n_iter = mcts_search(
        root_state=init_state,
        vlm=vlm,
        eval_llm=eval_models[0],  # Use first model for initial evaluation
        eval_llm_tokenizer=eval_tokenizers[0],
        question=question,
        answer=answer,
        generation_config=generation_config,
        n_iterations=n_iterations,
        c_puct=1.0,
        top_k=3
    )
    
    # If we have diverse solutions, use cross-model voting to rank them
    if len(diverse_solutions) >= 2:
        ranked_solutions = cross_model_voting(
            diverse_solutions, 
            eval_models, 
            eval_tokenizers, 
            question, 
            answer, 
            device
        )
        
        # Generate justifications for preference pairs
        justified_pairs = generate_justifications(
            ranked_solutions,
            eval_models[0],  # Use first model for justification generation
            eval_tokenizers[0],
            question,
            device
        )
        
        return root, steps, solution, diverse_solutions, ranked_solutions, justified_pairs, n_iter
    else:
        return root, steps, solution, diverse_solutions, [], [], n_iter