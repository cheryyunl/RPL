import torch
from PIL import Image
import io
from tqdm import tqdm

class DPOTrainer:
    def __init__(self, model, optimizer, tokenizer, processor, beta=0.1, device="cuda"):
        self.model = model
        self.optimizer = optimizer
        self.tokenizer = tokenizer
        self.processor = processor
        self.beta = beta
        self.device = device
        
    def compute_dpo_loss(self, chosen_logps, rejected_logps):
        """Compute DPO loss based on log probabilities of chosen and rejected responses."""
        losses = -torch.log(torch.sigmoid(self.beta * (chosen_logps - rejected_logps)))
        return losses.mean()
    
    def train_step(self, batch):
        """Perform a single training step using DPO."""
        self.optimizer.zero_grad()
        
        # Process inputs for both chosen and rejected responses
        chosen_inputs = self.processor(
            text=[batch["prompt"] for _ in range(len(batch["chosen"]))],
            images=[Image.open(io.BytesIO(img)) for img in batch["image"]],
            padding=True,
            return_tensors="pt",
        ).to(self.device)
        
        rejected_inputs = self.processor(
            text=[batch["prompt"] for _ in range(len(batch["rejected"]))],
            images=[Image.open(io.BytesIO(img)) for img in batch["image"]],
            padding=True,
            return_tensors="pt",
        ).to(self.device)
        
        # Get chosen and rejected log probabilities
        chosen_logps = self.get_logps(chosen_inputs, batch["chosen"])
        rejected_logps = self.get_logps(rejected_inputs, batch["rejected"])
        
        # Compute DPO loss
        loss = self.compute_dpo_loss(chosen_logps, rejected_logps)
        
        # Backward pass and optimization
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def get_logps(self, inputs, responses):
        """Get log probabilities of responses given inputs."""
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
        
        # Process response tokens
        response_tokens = self.tokenizer(responses, return_tensors="pt", padding=True).to(self.device)
        
        # Calculate log probabilities
        logps = []
        for i in range(len(responses)):
            logp = 0
            for j in range(len(response_tokens.input_ids[i]) - 1):
                token_id = response_tokens.input_ids[i][j+1]
                token_logits = logits[i, j, :]
                token_logp = torch.log_softmax(token_logits, dim=0)[token_id]
                logp += token_logp
            logps.append(logp)
        
        return torch.tensor(logps, device=self.device)
    
    def train(self, train_dataloader, epochs, save_interval=1):
        """Train the model using DPO for multiple epochs."""
        self.model.train()
        
        for epoch in range(epochs):
            total_loss = 0
            for batch in tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
                loss = self.train_step(batch)
                total_loss += loss
            
            print(f"Epoch {epoch+1}/{epochs}, Average Loss: {total_loss/len(train_dataloader)}")

            # Save the model every save_interval epochs
            if (epoch + 1) % save_interval == 0:
                self.model.save_pretrained(f"dpo_model_epoch_{epoch+1}")
                self.tokenizer.save_pretrained(f"dpo_tokenizer_epoch_{epoch+1}")
                self.processor.save_pretrained(f"dpo_processor_epoch_{epoch+1}")
                self.optimizer.save_pretrained(f"dpo_optimizer_epoch_{epoch+1}")
