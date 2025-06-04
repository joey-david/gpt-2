"""
Data loading and preprocessing utilities for GPT-2 training
"""
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import List, Tuple, Optional
import random


class TextDataset(Dataset):
    """Dataset for autoregressive language modeling"""
    
    def __init__(
        self,
        text: str,
        tokenizer,
        block_size: int = 1024,
        stride: int = None
    ):
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.stride = stride or block_size // 2
        
        # Tokenize the entire text
        self.tokens = tokenizer.encode(text)
        print(f"Total tokens: {len(self.tokens)}")
        
        # Create overlapping chunks
        self.examples = []
        for i in range(0, len(self.tokens) - block_size + 1, self.stride):
            chunk = self.tokens[i:i + block_size]
            if len(chunk) == block_size:
                self.examples.append(chunk)
        
        print(f"Created {len(self.examples)} training examples")
    
    def __len__(self) -> int:
        return len(self.examples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        tokens = self.examples[idx]
        
        # Input is all tokens except the last one
        # Target is all tokens except the first one (shifted by 1)
        input_ids = torch.tensor(tokens[:-1], dtype=torch.long)
        target_ids = torch.tensor(tokens[1:], dtype=torch.long)
        
        return input_ids, target_ids


class CharLevelDataset(Dataset):
    """Character-level dataset with sliding window approach"""
    
    def __init__(self, text: str, seq_length: int = 256):
        self.text = text
        self.seq_length = seq_length
        
        # Create character-to-index mapping
        self.chars = sorted(list(set(text)))
        self.char_to_idx = {ch: i for i, ch in enumerate(self.chars)}
        self.idx_to_char = {i: ch for i, ch in enumerate(self.chars)}
        self.vocab_size = len(self.chars)
        
        print(f"Character vocabulary size: {self.vocab_size}")
        print(f"Sample characters: {self.chars[:20]}")
        
    def __len__(self) -> int:
        return len(self.text) - self.seq_length
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Get sequence of characters
        chunk = self.text[idx:idx + self.seq_length + 1]
        
        # Convert to indices
        input_seq = [self.char_to_idx[ch] for ch in chunk[:-1]]
        target_seq = [self.char_to_idx[ch] for ch in chunk[1:]]
        
        return torch.tensor(input_seq, dtype=torch.long), torch.tensor(target_seq, dtype=torch.long)


def create_data_loaders(
    text: str,
    tokenizer,
    batch_size: int = 32,
    block_size: int = 512,
    train_split: float = 0.9,
    num_workers: int = 4
) -> Tuple[DataLoader, DataLoader]:
    """Create train and validation data loaders"""
    
    # Split text into train and validation
    split_idx = int(len(text) * train_split)
    train_text = text[:split_idx]
    val_text = text[split_idx:]
    
    print(f"Train text length: {len(train_text)}")
    print(f"Validation text length: {len(val_text)}")
    
    # Create datasets
    train_dataset = TextDataset(train_text, tokenizer, block_size)
    val_dataset = TextDataset(val_text, tokenizer, block_size)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader


def collate_batch(batch: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
    """Custom collate function for variable length sequences"""
    inputs, targets = zip(*batch)
    
    # Pad sequences to same length
    max_len = max(len(seq) for seq in inputs)
    
    padded_inputs = []
    padded_targets = []
    
    for inp, tgt in zip(inputs, targets):
        # Pad with zeros (or a special padding token)
        pad_len = max_len - len(inp)
        padded_inp = torch.cat([inp, torch.zeros(pad_len, dtype=torch.long)])
        padded_tgt = torch.cat([tgt, torch.zeros(pad_len, dtype=torch.long)])
        
        padded_inputs.append(padded_inp)
        padded_targets.append(padded_tgt)
    
    return torch.stack(padded_inputs), torch.stack(padded_targets)


class DataProcessor:
    """Utility class for text preprocessing"""
    
    @staticmethod
    def clean_text(text: str) -> str:
        """Basic text cleaning"""
        # Remove Project Gutenberg headers/footers
        lines = text.split('\n')
        start_idx = 0
        end_idx = len(lines)
        
        for i, line in enumerate(lines):
            if "START OF" in line and "PROJECT GUTENBERG" in line:
                start_idx = i + 1
                break
        
        for i in range(len(lines) - 1, -1, -1):
            if "END OF" in lines[i] and "PROJECT GUTENBERG" in lines[i]:
                end_idx = i
                break
        
        cleaned_lines = lines[start_idx:end_idx]
        cleaned_text = '\n'.join(cleaned_lines)
        
        # Basic cleaning
        cleaned_text = cleaned_text.replace('\r\n', '\n')
        cleaned_text = cleaned_text.replace('\r', '\n')
        
        # Remove excessive whitespace
        lines = cleaned_text.split('\n')
        cleaned_lines = []
        for line in lines:
            line = line.strip()
            if line:  # Only keep non-empty lines
                cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines)
    
    @staticmethod
    def get_text_statistics(text: str) -> dict:
        """Get basic statistics about the text"""
        stats = {
            'total_chars': len(text),
            'unique_chars': len(set(text)),
            'lines': len(text.split('\n')),
            'words': len(text.split()),
            'vocab': sorted(list(set(text)))
        }
        return stats


def sample_batch(data_loader: DataLoader, device: str = 'cpu') -> Tuple[torch.Tensor, torch.Tensor]:
    """Sample a single batch from data loader"""
    batch = next(iter(data_loader))
    inputs, targets = batch
    return inputs.to(device), targets.to(device)


def estimate_loss(model, data_loader: DataLoader, device: str, max_batches: int = 100) -> float:
    """Estimate average loss on a dataset"""
    model.eval()
    total_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        for i, (inputs, targets) in enumerate(data_loader):
            if i >= max_batches:
                break
            
            inputs, targets = inputs.to(device), targets.to(device)
            logits = model(inputs)
            
            # Reshape for cross entropy loss
            logits = logits.view(-1, logits.size(-1))
            targets = targets.view(-1)
            
            loss = torch.nn.functional.cross_entropy(logits, targets, ignore_index=-1)
            total_loss += loss.item()
            num_batches += 1
    
    model.train()
    return total_loss / num_batches if num_batches > 0 else float('inf')
