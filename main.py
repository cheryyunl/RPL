import argparse
import torch
import pandas as pd
from tqdm import tqdm
from transformers import GenerationConfig, Qwen2_5_VLForConditionalGeneration, AutoModelForCausalLM, AutoProcessor, AutoTokenizer
from torch.utils.data import DataLoader
from utils import get_chunk, split_list
from model import VisionLanguageModel
from rpl_solver import solve_math_reasoning_vlm_with_rpl
from dpo_trainer import DPOTrainer
from data_processing import prepare_preference_dataset
from prompt_templates import few_shot_cot_prompt

def main(args):
    device = f"cuda:{args.gpu_id}"
    generation_config = GenerationConfig(
        temperature=0.5,
        do_sample=True,
        top_p=0.9,
    )

    # Load main model for reasoning
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_id, torch_dtype=torch.float16, device_map=device
    )
    processor = AutoProcessor.from_pretrained(args.model_id)

    # Load multiple evaluation models for cross-model voting
    eval_models = []
    eval_tokenizers = []
    
    # Primary evaluation model
    eval_model = AutoModelForCausalLM.from_pretrained(
        args.eval_model_name,
        torch_dtype="auto",
        device_map=device
    )
    eval_tokenizer = AutoTokenizer.from_pretrained(args.eval_model_name)
    eval_models.append(eval_model)
    eval_tokenizers.append(eval_tokenizer)
    
    # Add additional evaluation models for cross-model voting
    if args.additional_eval_models:
        for model_name in args.additional_eval_models:
            add_model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype="auto",
                device_map=device
            )
            add_tokenizer = AutoTokenizer.from_pretrained(model_name)
            eval_models.append(add_model)
            eval_tokenizers.append(add_tokenizer)
    
    # Load data
    df = pd.read_parquet(args.data_pths, engine='pyarrow')
    datas = df.to_dict(orient='records')
    data_chunk = get_chunk(datas, args.num_chunks, args.chunk_idx)
    
    # Results storage
    final_response = []
    all_preference_pairs = []
    
    # Phase 1: Generate diverse reasoning paths and create preference pairs
    for data in tqdm(data_chunk, desc="Generating Reasoning Paths"):
        image_data = data['image']
        question = data['problem'].split('<image>')[1]
        answer = data['answer']
        text_prompt = few_shot_cot_prompt + '{}'.format(question)

        # Enhanced solve function with RPL components
        root, solution_steps, solution, diverse_solutions, ranked_solutions, justified_pairs, n_iter = solve_math_reasoning_vlm_with_rpl(
            image_data=image_data,
            text_prompt=text_prompt,
            model=model,
            generation_config=generation_config,
            processor=processor,
            eval_models=eval_models,
            eval_tokenizers=eval_tokenizers,
            question=question,
            answer=answer,
            n_iterations=args.max_num_iterations,
            device=device
        )

        if solution is not None:
            data['solution'] = ''.join(solution.solution_steps)
            data['iters'] = n_iter
            data['num_diverse_solutions'] = len(diverse_solutions)
            final_response.append(data)
            
        # Store preference pairs for DPO training
        if justified_pairs:
            for pair in justified_pairs:
                pair_with_image = pair.copy()
                pair_with_image["image"] = image_data
                all_preference_pairs.append(pair_with_image)
    
    # Save initial results
    df = pd.DataFrame(final_response)
    df.to_parquet(args.output_file, index=False, engine='pyarrow')
    
    # Phase 2: DPO training if preference pairs were generated
    if len(all_preference_pairs) > 0 and args.do_train:
        print(f"Generated {len(all_preference_pairs)} preference pairs for training")
        
        # Prepare dataset for DPO
        preference_dataset = prepare_preference_dataset(
            all_preference_pairs,
            [pair["image"] for pair in all_preference_pairs]
        )
        
        # Setup DPO training
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
        dpo_trainer = DPOTrainer(
            model=model,
            optimizer=optimizer,
            tokenizer=processor.tokenizer,
            processor=processor,
            beta=args.beta,
            device=device
        )
        
        # Create DataLoader for training
        train_dataloader = DataLoader(
            preference_dataset,
            batch_size=args.batch_size,
            shuffle=True
        )
        
        # Train with DPO
        print("Starting DPO training...")
        dpo_trainer.train(train_dataloader, args.epochs)
        
        # Save the fine-tuned model
        model.save_pretrained(args.output_model_dir)
        processor.save_pretrained(args.output_model_dir)
        print(f"Model saved to {args.output_model_dir}")
        
        # Save preference dataset for future use
        preference_df = pd.DataFrame(all_preference_pairs)
        preference_df.to_parquet(args.preference_data_file, index=False, engine='pyarrow')
        print(f"Preference data saved to {args.preference_data_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--eval_model_name", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--additional_eval_models", type=str, nargs='+', default=[])
    parser.add_argument("--data_pths", type=str, default="None")
    parser.add_argument("--output_file", type=str, default="answer.jsonl")
    parser.add_argument("--preference_data_file", type=str, default="preference_data.parquet")
    parser.add_argument("--output_model_dir", type=str, default="rpl_trained_model")
    parser.add_argument("--max_num_iterations", type=int, default=50)
    parser.add_argument("--num_chunks", type=int, default=1)
    parser.add_argument("--chunk_idx", type=int, default=0)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--do_train", action="store_true", help="Whether to perform DPO training")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--beta", type=float, default=0.1, help="DPO beta parameter")
    
    args = parser.parse_args()

    main(args)