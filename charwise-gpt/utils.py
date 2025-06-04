"""
Configuration and utility functions for GPT-2 implementation
"""
import json
import os
import torch
import random
import numpy as np
from typing import Dict, Any


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_config(config: Dict[str, Any], filepath: str):
    """Save configuration to JSON file"""
    with open(filepath, 'w') as f:
        json.dump(config, f, indent=4)


def load_config(filepath: str) -> Dict[str, Any]:
    """Load configuration from JSON file"""
    with open(filepath, 'r') as f:
        return json.load(f)


def count_parameters(model):
    """Count total and trainable parameters in model"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def get_model_size_mb(model):
    """Get model size in MB"""
    param_size = 0
    buffer_size = 0
    
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_mb = (param_size + buffer_size) / 1024**2
    return size_mb


def format_time(seconds):
    """Format time in human readable format"""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds//60:.0f}m {seconds%60:.0f}s"
    else:
        return f"{seconds//3600:.0f}h {(seconds%3600)//60:.0f}m {seconds%60:.0f}s"


class ModelSaver:
    """Utility class for saving and loading models"""
    
    def __init__(self, save_dir: str = "checkpoints"):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
    
    def save_model(self, model, tokenizer, config, filename: str, **kwargs):
        """Save complete model state"""
        filepath = os.path.join(self.save_dir, filename)
        
        save_dict = {
            'model_state_dict': model.state_dict(),
            'config': config,
            'vocab_size': tokenizer.vocab_size,
            **kwargs
        }
        
        torch.save(save_dict, filepath)
        
        # Also save tokenizer separately
        tokenizer_path = filepath.replace('.pt', '_tokenizer.json')
        tokenizer.save(tokenizer_path)
        
        print(f"Model saved to {filepath}")
        print(f"Tokenizer saved to {tokenizer_path}")
    
    def load_model(self, model_class, filename: str):
        """Load complete model state"""
        filepath = os.path.join(self.save_dir, filename)
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        checkpoint = torch.load(filepath, map_location='cpu')
        
        # Create model with saved config
        config = checkpoint['config']
        model = model_class(**config)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        print(f"Model loaded from {filepath}")
        return model, checkpoint


def print_model_info(model, tokenizer):
    """Print detailed model information"""
    total_params, trainable_params = count_parameters(model)
    model_size = get_model_size_mb(model)
    
    print("=" * 60)
    print("MODEL INFORMATION")
    print("=" * 60)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Model size: {model_size:.2f} MB")
    print(f"Vocabulary size: {tokenizer.vocab_size:,}")
    print(f"Architecture: GPT-2")
    print("=" * 60)


# Training hyperparameters for different model sizes
TRAINING_CONFIGS = {
    'small': {
        'batch_size': 32,
        'learning_rate': 3e-4,
        'weight_decay': 0.1,
        'warmup_steps': 1000,
        'max_steps': 10000,
        'eval_interval': 500,
        'save_interval': 1000,
    },
    'medium': {
        'batch_size': 16,
        'learning_rate': 2e-4,
        'weight_decay': 0.1,
        'warmup_steps': 2000,
        'max_steps': 20000,
        'eval_interval': 1000,
        'save_interval': 2000,
    },
    'large': {
        'batch_size': 8,
        'learning_rate': 1e-4,
        'weight_decay': 0.1,
        'warmup_steps': 4000,
        'max_steps': 40000,
        'eval_interval': 2000,
        'save_interval': 4000,
    }
}
