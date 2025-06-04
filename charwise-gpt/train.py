"""
Training script for GPT-2 model on Charles Dickens corpus
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
import os
from typing import Dict, List, Tuple
import time

# Local imports
from model import GPT2Model, GPT2Config, GPT2_CONFIGS
from tokenizer import CharacterTokenizer
from data_utils import TextDataset, create_data_loaders, DataProcessor, estimate_loss


class GPT2Trainer:
    """Complete training class for GPT-2"""
    
    def __init__(
        self,
        model: GPT2Model,
        tokenizer: CharacterTokenizer,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: str = 'cuda',
        learning_rate: float = 3e-4,
        weight_decay: float = 0.1,
        betas: Tuple[float, float] = (0.9, 0.95)
    ):
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        # Optimizer (AdamW as used in GPT-2)training exa
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=betas
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, 
            T_max=len(train_loader) * 10,  # Assuming 10 epochs
            eta_min=learning_rate * 0.1
        )
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss(ignore_index=-1)
        
        # Training metrics
        self.train_losses = []
        self.val_losses = []
        self.learning_rates = []
        
    def train_epoch(self) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc="Training")
        for batch_idx, (inputs, targets) in enumerate(pbar):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            logits = self.model(inputs)
            
            # Reshape for loss calculation
            logits = logits.view(-1, logits.size(-1))
            targets = targets.view(-1)
            
            loss = self.criterion(logits, targets)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping (important for transformer training)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            self.scheduler.step()
            
            # Update metrics
            total_loss += loss.item()
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'lr': f'{self.scheduler.get_last_lr()[0]:.2e}'
            })
        
        return total_loss / num_batches
    
    def validate(self) -> float:
        """Validate the model"""
        return estimate_loss(self.model, self.val_loader, self.device, max_batches=50)
    
    def train(self, num_epochs: int, save_every: int = 5, generate_every: int = 2):
        """Complete training loop"""
        print(f"Starting training for {num_epochs} epochs")
        print(f"Device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        best_val_loss = float('inf')
        
        for epoch in range(num_epochs):
            start_time = time.time()
            
            # Training
            train_loss = self.train_epoch()
            
            # Validation
            val_loss = self.validate()
            
            # Record metrics
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.learning_rates.append(self.scheduler.get_last_lr()[0])
            
            epoch_time = time.time() - start_time
            
            print(f"Epoch {epoch+1}/{num_epochs}")
            print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            print(f"Time: {epoch_time:.2f}s | LR: {self.scheduler.get_last_lr()[0]:.2e}")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_checkpoint(f'best_model.pt', epoch, train_loss, val_loss)
                print("✓ Saved best model")
            
            # Regular checkpointing
            if (epoch + 1) % save_every == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch+1}.pt', epoch, train_loss, val_loss)
            
            # Generate sample text
            if (epoch + 1) % generate_every == 0:
                print("\n--- Sample Generation ---")
                self.generate_sample("The old man ")
                print("------------------------\n")
        
        print("Training completed!")
        self.plot_training_curves()
    
    def generate_sample(self, prompt: str = "It was", max_length: int = 200):
        """Generate sample text"""
        self.model.eval()
        
        # Encode prompt
        input_ids = torch.tensor([self.tokenizer.encode(prompt)], device=self.device)
        
        with torch.no_grad():
            generated = self.model.generate(
                input_ids,
                max_length=max_length,
                temperature=0.8,
                top_k=50,
                top_p=0.9
            )
        
        # Decode generated text
        generated_text = self.tokenizer.decode(generated[0].tolist())
        print(f"Generated: {generated_text}")
        
        self.model.train()
        return generated_text
    
    def save_checkpoint(self, filename: str, epoch: int, train_loss: float, val_loss: float):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'config': {
                'vocab_size': self.model.vocab_size,
                'd_model': self.model.d_model,
                'max_seq_len': self.model.max_seq_len
            }
        }
        torch.save(checkpoint, filename)
    
    def load_checkpoint(self, filename: str):
        """Load model checkpoint"""
        checkpoint = torch.load(filename, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
        return checkpoint['epoch']
    
    def plot_training_curves(self):
        """Plot training and validation loss curves"""
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(self.train_losses, label='Train Loss')
        plt.plot(self.val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Curves')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(1, 2, 2)
        plt.plot(self.learning_rates)
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.title('Learning Rate Schedule')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('training_curves.png', dpi=300, bbox_inches='tight')
        plt.show()


def main():
    """Main training function"""
    # Configuration
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load and preprocess text
    print("Loading Charles Dickens corpus...")
    with open("charwise-gpt/dickens/combined.txt", 'r', encoding='utf-8') as f:
        raw_text = f.read()
    
    # Clean the text
    processor = DataProcessor()
    text = processor.clean_text(raw_text)
    stats = processor.get_text_statistics(text)
    
    print(f"Text statistics:")
    print(f"  Total characters: {stats['total_chars']:,}")
    print(f"  Unique characters: {stats['unique_chars']}")
    print(f"  Lines: {stats['lines']:,}")
    print(f"  Words: {stats['words']:,}")
    
    # Create tokenizer
    print("\nBuilding tokenizer...")
    tokenizer = CharacterTokenizer()
    tokenizer.build_vocab(text)
    tokenizer.save('tokenizer.json')
    
    # Model configuration (small model for character-level)
    config = GPT2Config(
        vocab_size=tokenizer.vocab_size,
        d_model=384,
        n_heads=6,
        n_layers=6,
        d_ff=1536,
        max_seq_len=512,
        dropout=0.1
    )
    
    # Create model
    print(f"\nCreating GPT-2 model...")
    model = GPT2Model(
        vocab_size=config.vocab_size,
        d_model=config.d_model,
        n_heads=config.n_heads,
        n_layers=config.n_layers,
        d_ff=config.d_ff,
        max_seq_len=config.max_seq_len,
        dropout=config.dropout
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create data loaders
    print("\nPreparing data loaders...")
    train_loader, val_loader = create_data_loaders(
        text=text,
        tokenizer=tokenizer,
        batch_size=8,
        block_size=config.max_seq_len,
        train_split=0.9,
        num_workers=4
    )
    
    # Create trainer
    trainer = GPT2Trainer(
        model=model,
        tokenizer=tokenizer,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        learning_rate=5e-4
    )
    
    # Train the model
    print("\nStarting training...")
    trainer.train(num_epochs=20, save_every=5, generate_every=2)
    
    # Final generation examples
    print("\n" + "="*50)
    print("FINAL GENERATION EXAMPLES")
    print("="*50)
    
    prompts = [
        "It was the best of times",
        "The old man",
        "London was",
        "Christmas morning"
    ]
    
    for prompt in prompts:
        print(f"\nPrompt: '{prompt}'")
        generated = trainer.generate_sample(prompt, max_length=300)
        print("-" * 50)


if __name__ == "__main__":
    main()

