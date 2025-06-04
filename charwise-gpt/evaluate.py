"""
Evaluation and analysis tools for the GPT-2 model
"""
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
import os
from typing import List, Dict, Tuple
from collections import Counter
import re

from model import GPT2Model
from tokenizer import CharacterTokenizer
from data_utils import TextDataset, estimate_loss


class ModelEvaluator:
    """Comprehensive model evaluation and analysis"""
    
    def __init__(self, model: GPT2Model, tokenizer: CharacterTokenizer, device: str = 'cuda'):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.eval()
    
    def calculate_perplexity(self, text: str, batch_size: int = 32, block_size: int = 512) -> float:
        """Calculate perplexity on given text"""
        dataset = TextDataset(text, self.tokenizer, block_size)
        total_loss = 0
        total_tokens = 0
        
        with torch.no_grad():
            for i in tqdm(range(0, len(dataset), batch_size), desc="Calculating perplexity"):
                batch_inputs = []
                batch_targets = []
                
                for j in range(i, min(i + batch_size, len(dataset))):
                    inputs, targets = dataset[j]
                    batch_inputs.append(inputs)
                    batch_targets.append(targets)
                
                if not batch_inputs:
                    break
                
                # Pad sequences to same length
                max_len = max(len(seq) for seq in batch_inputs)
                padded_inputs = []
                padded_targets = []
                
                for inp, tgt in zip(batch_inputs, batch_targets):
                    pad_len = max_len - len(inp)
                    padded_inp = torch.cat([inp, torch.zeros(pad_len, dtype=torch.long)])
                    # Use -100 for padding tokens (ignored in loss calculation)
                    padded_tgt = torch.cat([tgt, torch.full((pad_len,), -100, dtype=torch.long)])
                    
                    padded_inputs.append(padded_inp)
                    padded_targets.append(padded_tgt)
                
                inputs = torch.stack(padded_inputs).to(self.device)
                targets = torch.stack(padded_targets).to(self.device)
                
                logits = self.model(inputs)
                logits = logits.view(-1, logits.size(-1))
                targets = targets.view(-1)
                
                # Calculate loss only for non-padding tokens
                mask = targets != -100
                if mask.sum() > 0:
                    loss = F.cross_entropy(logits[mask], targets[mask])
                    total_loss += loss.item() * mask.sum().item()
                    total_tokens += mask.sum().item()
        
        avg_loss = total_loss / total_tokens if total_tokens > 0 else float('inf')
        perplexity = np.exp(avg_loss)
        return perplexity
    
    def analyze_attention_patterns(self, text: str, max_length: int = 100) -> Dict:
        """Analyze attention patterns in the model"""
        # Encode text
        tokens = self.tokenizer.encode(text[:max_length])
        input_ids = torch.tensor([tokens], device=self.device)
        
        # Hook to capture attention weights
        attention_weights = []
        
        def attention_hook(module, input, output):
            # This is a simplified version - in practice you'd need to modify
            # the attention module to return attention weights
            pass
        
        # Register hooks (simplified - would need proper implementation)
        hooks = []
        for layer in self.model.transformer_blocks:
            hook = layer.attention.register_forward_hook(attention_hook)
            hooks.append(hook)
        
        with torch.no_grad():
            _ = self.model(input_ids)
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
        
        return {"attention_weights": attention_weights}
    
    def analyze_token_predictions(self, text: str, top_k: int = 10) -> Dict:
        """Analyze model's top predictions for each token"""
        tokens = self.tokenizer.encode(text)
        predictions = []
        
        with torch.no_grad():
            for i in range(1, len(tokens)):
                context = torch.tensor([tokens[:i]], device=self.device)
                logits = self.model(context)
                next_token_logits = logits[0, -1, :]
                
                # Get top-k predictions
                top_k_logits, top_k_indices = torch.topk(next_token_logits, top_k)
                top_k_probs = F.softmax(top_k_logits, dim=0)
                
                # Convert to characters
                top_k_chars = [self.tokenizer.decode([idx.item()]) for idx in top_k_indices]
                actual_char = self.tokenizer.decode([tokens[i]])
                
                predictions.append({
                    'position': i,
                    'actual': actual_char,
                    'context': self.tokenizer.decode(tokens[:i]),
                    'top_predictions': list(zip(top_k_chars, top_k_probs.tolist()))
                })
        
        return predictions
    
    def generate_diverse_samples(
        self,
        prompts: List[str],
        num_samples: int = 3,
        max_length: int = 200,
        temperatures: List[float] = [0.5, 0.8, 1.2]
    ) -> Dict:
        """Generate diverse samples with different temperatures"""
        results = {}
        
        for prompt in prompts:
            results[prompt] = {}
            
            for temp in temperatures:
                samples = []
                for _ in range(num_samples):
                    input_ids = torch.tensor([self.tokenizer.encode(prompt)], device=self.device)
                    
                    with torch.no_grad():
                        generated = self.model.generate(
                            input_ids,
                            max_length=max_length,
                            temperature=temp,
                            top_k=50,
                            top_p=0.9
                        )
                    
                    generated_text = self.tokenizer.decode(generated[0].tolist())
                    samples.append(generated_text)
                
                results[prompt][f'temp_{temp}'] = samples
        
        return results
    
    def calculate_text_statistics(self, generated_texts: List[str]) -> Dict:
        """Calculate various statistics for generated text"""
        stats = {
            'avg_length': np.mean([len(text) for text in generated_texts]),
            'unique_chars': len(set(''.join(generated_texts))),
            'repetition_ratio': [],
            'word_counts': Counter(),
            'char_counts': Counter()
        }
        
        for text in generated_texts:
            # Character frequency
            stats['char_counts'].update(text)
            
            # Word frequency (simple split)
            words = re.findall(r'\b\w+\b', text.lower())
            stats['word_counts'].update(words)
            
            # Repetition ratio (4-gram repetition)
            n = 4
            ngrams = [text[i:i+n] for i in range(len(text)-n+1)]
            if ngrams:
                unique_ngrams = len(set(ngrams))
                repetition = 1 - (unique_ngrams / len(ngrams))
                stats['repetition_ratio'].append(repetition)
        
        stats['avg_repetition'] = np.mean(stats['repetition_ratio']) if stats['repetition_ratio'] else 0
        return stats
    
    def run_comprehensive_evaluation(self, test_text: str, output_dir: str = "evaluation_results"):
        """Run comprehensive evaluation and save results"""
        os.makedirs(output_dir, exist_ok=True)
        
        print("Running comprehensive evaluation...")
        
        # 1. Calculate perplexity
        print("Calculating perplexity...")
        perplexity = self.calculate_perplexity(test_text)
        print(f"Perplexity: {perplexity:.2f}")
        
        # 2. Generate diverse samples
        print("Generating diverse samples...")
        prompts = [
            "It was a dark and stormy",
            "The old gentleman",
            "Christmas morning brought",
            "In the heart of London",
            "The child looked up"
        ]
        
        diverse_samples = self.generate_diverse_samples(prompts)
        
        # 3. Analyze generated text statistics
        print("Analyzing text statistics...")
        all_generated = []
        for prompt_results in diverse_samples.values():
            for temp_results in prompt_results.values():
                all_generated.extend(temp_results)
        
        text_stats = self.calculate_text_statistics(all_generated)
        
        # 4. Token prediction analysis
        print("Analyzing token predictions...")
        sample_text = test_text[:200]  # First 200 characters for analysis
        prediction_analysis = self.analyze_token_predictions(sample_text)
        
        # Save results
        results = {
            'perplexity': perplexity,
            'diverse_samples': diverse_samples,
            'text_statistics': {
                'avg_length': text_stats['avg_length'],
                'unique_chars': text_stats['unique_chars'],
                'avg_repetition': text_stats['avg_repetition'],
                'top_words': dict(text_stats['word_counts'].most_common(20)),
                'top_chars': dict(text_stats['char_counts'].most_common(20))
            },
            'sample_predictions': prediction_analysis[:10]  # First 10 for brevity
        }
        
        # Save to JSON
        with open(os.path.join(output_dir, 'evaluation_results.json'), 'w') as f:
            json.dump(results, f, indent=2)
        
        # Create visualization
        self.create_evaluation_plots(results, output_dir)
        
        print(f"Evaluation completed! Results saved to {output_dir}/")
        return results
    
    def create_evaluation_plots(self, results: Dict, output_dir: str):
        """Create visualization plots for evaluation results"""
        plt.style.use('default')
        
        # 1. Character frequency plot
        char_counts = results['text_statistics']['top_chars']
        if char_counts:
            plt.figure(figsize=(12, 6))
            chars = list(char_counts.keys())[:15]
            counts = [char_counts[c] for c in chars]
            
            plt.bar(range(len(chars)), counts)
            plt.xlabel('Characters')
            plt.ylabel('Frequency')
            plt.title('Top Character Frequencies in Generated Text')
            plt.xticks(range(len(chars)), [repr(c) for c in chars], rotation=45)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'char_frequency.png'), dpi=300)
            plt.close()
        
        # 2. Word frequency plot
        word_counts = results['text_statistics']['top_words']
        if word_counts:
            plt.figure(figsize=(12, 6))
            words = list(word_counts.keys())[:15]
            counts = [word_counts[w] for w in words]
            
            plt.bar(range(len(words)), counts)
            plt.xlabel('Words')
            plt.ylabel('Frequency')
            plt.title('Top Word Frequencies in Generated Text')
            plt.xticks(range(len(words)), words, rotation=45)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'word_frequency.png'), dpi=300)
            plt.close()
        
        # 3. Temperature comparison
        if 'diverse_samples' in results:
            plt.figure(figsize=(10, 6))
            
            # Collect lengths for different temperatures
            temp_lengths = {}
            for prompt, temp_results in results['diverse_samples'].items():
                for temp_key, samples in temp_results.items():
                    temp = float(temp_key.split('_')[1])
                    if temp not in temp_lengths:
                        temp_lengths[temp] = []
                    temp_lengths[temp].extend([len(s) for s in samples])
            
            temps = sorted(temp_lengths.keys())
            avg_lengths = [np.mean(temp_lengths[t]) for t in temps]
            std_lengths = [np.std(temp_lengths[t]) for t in temps]
            
            plt.errorbar(temps, avg_lengths, yerr=std_lengths, marker='o', capsize=5)
            plt.xlabel('Temperature')
            plt.ylabel('Average Generated Length')
            plt.title('Generation Length vs Temperature')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'temperature_analysis.png'), dpi=300)
            plt.close()
        
        print("Evaluation plots saved!")


def main():
    """Main evaluation function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate GPT-2 model")
    parser.add_argument('--model', type=str, default='best_model.pt',
                       help='Path to trained model')
    parser.add_argument('--tokenizer', type=str, default='tokenizer.json',
                       help='Path to tokenizer')
    parser.add_argument('--test_text', type=str, default='charwise-gpt/dickens/combined.txt',
                       help='Path to test text file')
    parser.add_argument('--output_dir', type=str, default='evaluation_results',
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    # Load model and tokenizer
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    tokenizer = CharacterTokenizer()
    tokenizer.load(args.tokenizer)
    
    checkpoint = torch.load(args.model, map_location=device)
    config = checkpoint['config']
    
    model = GPT2Model(
        vocab_size=config['vocab_size'],
        d_model=config.get('d_model', 384),
        n_heads=config.get('n_heads', 6),
        n_layers=config.get('n_layers', 6),
        d_ff=config.get('d_ff', 1536),
        max_seq_len=config.get('max_seq_len', 512)
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    
    # Load test text
    with open(args.test_text, 'r', encoding='utf-8') as f:
        test_text = f.read()
    
    # Take a subset for evaluation (last 10% for testing)
    test_text = test_text[-len(test_text)//10:]
    
    # Run evaluation
    evaluator = ModelEvaluator(model, tokenizer, device)
    results = evaluator.run_comprehensive_evaluation(test_text, args.output_dir)
    
    print(f"\nEvaluation Summary:")
    print(f"Perplexity: {results['perplexity']:.2f}")
    print(f"Average generation length: {results['text_statistics']['avg_length']:.1f}")
    print(f"Unique characters in generated text: {results['text_statistics']['unique_chars']}")
    print(f"Average repetition ratio: {results['text_statistics']['avg_repetition']:.3f}")


if __name__ == "__main__":
    main()
