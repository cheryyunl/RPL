import torch
from PIL import Image
import io

class State:
    def __init__(self, image_feat, text_context, solution_steps=None):
        self.image_feat = image_feat
        self.text_context = text_context
        self.solution_steps = solution_steps if solution_steps else []
        self.is_terminal = False

    def copy(self):
        new_state = State(
            image_feat=self.image_feat,
            text_context=self.text_context,
            solution_steps=self.solution_steps.copy()
        )
        new_state.is_terminal = self.is_terminal
        return new_state

    def __repr__(self):
        return f"<State steps={len(self.solution_steps)}, terminal={self.is_terminal}>"

class Action:
    def __init__(self, text):
        self.text = text

    def __repr__(self):
        return f"<Action: {self.text}>"

class VisionLanguageModel:
    def __init__(self, model, processor):
        self.model = model
        self.processor = processor

    def _run_vlm(self, image_feat, text_context, generation_config, history=None):
        prompt = text_context
        message = [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image"},
            ],
        }]
        if history:
            message.append({
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "".join(history)}, ],
            })

        text = self.processor.apply_chat_template(
            message, tokenize=False, add_generation_prompt=True
        )[:-32]
        image_inputs = Image.open(io.BytesIO(image_feat))
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.model.device)
        question_input_length = inputs['input_ids'].shape[1]

        generated_ids = self.model.generate(**inputs, generation_config=generation_config, stop_strings=['<end>'],
                                       max_new_tokens=2048, tokenizer=self.processor.tokenizer)
        output = self.processor.decode(
            generated_ids[0][question_input_length:], skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        return output

    def propose_actions(self, state, generation_config, top_k=3):
        actions = []
        for i in range(top_k):
            llama_output = self._run_vlm(
                image_feat=state.image_feat,
                text_context=state.text_context,
                generation_config=generation_config,
                history=state.solution_steps
            )
            action_text = llama_output
            prob = 1.0 / top_k
            actions.append((Action(action_text), prob))
        return actions

    def transition(self, state, action):
        next_state = state.copy()
        next_state.solution_steps.append(action.text)

        if len(next_state.solution_steps) >= 10 or "Final Answer: " in next_state.solution_steps[-1]:
            next_state.is_terminal = True
        return next_state

    def evaluate_terminal_state(self, state, eval_llm, eval_llm_tokenizer, question, answer):
        if state.is_terminal:
            from prompt_templates import eval_prompt_template
            
            simulation_response = "".join(state.solution_steps)
            prompt = eval_prompt_template.format(question, answer, simulation_response)

            messages = [
                {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ]
            text = eval_llm_tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            model_inputs = eval_llm_tokenizer([text], return_tensors="pt").to(eval_llm.device)

            generated_ids = eval_llm.generate(
                **model_inputs,
                max_new_tokens=512
            )
            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]

            response = eval_llm_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

            if 'true' in response.split('.')[0].lower():
                return 1.0
            else:
                return 0.0
        return 0.0