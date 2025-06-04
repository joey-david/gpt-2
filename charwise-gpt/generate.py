"""
Interactive text generation script for the trained GPT-2 model
"""
import torch
import argparse
from model import GPT2Model
from tokenizer import CharacterTokenizer
from utils import ModelSaver, set_seed
import os


class TextGenerator:
    """Interactive text generator using trained GPT-2 model"""
    
    def __init__(self, model_path: str, tokenizer_path: str, device: str = 'auto'):
        self.device = device if device != 'auto' else ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load tokenizer
        self.tokenizer = CharacterTokenizer()
        self.tokenizer.load(tokenizer_path)
        
        # Load model
        checkpoint = torch.load(model_path, map_location=self.device)
        config = checkpoint['config']
        
        self.model = GPT2Model(
            vocab_size=config['vocab_size'],
            d_model=config.get('d_model', 384),
            n_heads=config.get('n_heads', 6),
            n_layers=config.get('n_layers', 6),
            d_ff=config.get('d_ff', 1536),
            max_seq_len=config.get('max_seq_len', 512),
            dropout=0.0  # No dropout during inference
        )
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        print(f"Model loaded on {self.device}")
        print(f"Vocabulary size: {self.tokenizer.vocab_size}")
    
    def generate(
        self,
        prompt: str,
        max_length: int = 200,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.9,
        num_samples: int = 1
    ) -> list:
        """Generate text from prompt"""
        results = []
        
        for _ in range(num_samples):
            # Encode prompt
            input_ids = torch.tensor([self.tokenizer.encode(prompt)], device=self.device)
            
            with torch.no_grad():
                generated = self.model.generate(
                    input_ids,
                    max_length=max_length,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p
                )
            
            # Decode generated text
            generated_text = self.tokenizer.decode(generated[0].tolist())
            results.append(generated_text)
        
        return results
    
    def interactive_mode(self):
        """Run interactive generation mode"""
        print("\n" + "="*60)
        print("DICKENS GPT-2 INTERACTIVE TEXT GENERATOR")
        print("="*60)
        print("Enter prompts to generate text in the style of Charles Dickens")
        print("Commands:")
        print("  /quit or /exit - Exit the program")
        print("  /help - Show this help message")
        print("  /settings - Show current generation settings")
        print("  /temp <value> - Set temperature (0.1-2.0)")
        print("  /length <value> - Set max generation length")
        print("  /samples <value> - Set number of samples to generate")
        print("="*60)
        
        # Default settings
        temperature = 0.8
        max_length = 300
        num_samples = 1
        top_k = 50
        top_p = 0.9
        
        while True:
            try:
                user_input = input("\nEnter prompt (or command): ").strip()
                
                if not user_input:
                    continue
                
                # Handle commands
                if user_input.startswith('/'):
                    command = user_input.lower()
                    
                    if command in ['/quit', '/exit']:
                        print("Goodbye!")
                        break
                    
                    elif command == '/help':
                        print("\nCommands:")
                        print("  /quit or /exit - Exit the program")
                        print("  /help - Show this help message")
                        print("  /settings - Show current generation settings")
                        print("  /temp <value> - Set temperature (0.1-2.0)")
                        print("  /length <value> - Set max generation length")
                        print("  /samples <value> - Set number of samples to generate")
                    
                    elif command == '/settings':
                        print(f"\nCurrent settings:")
                        print(f"  Temperature: {temperature}")
                        print(f"  Max length: {max_length}")
                        print(f"  Number of samples: {num_samples}")
                        print(f"  Top-k: {top_k}")
                        print(f"  Top-p: {top_p}")
                    
                    elif command.startswith('/temp '):
                        try:
                            new_temp = float(command.split()[1])
                            if 0.1 <= new_temp <= 2.0:
                                temperature = new_temp
                                print(f"Temperature set to {temperature}")
                            else:
                                print("Temperature must be between 0.1 and 2.0")
                        except (IndexError, ValueError):
                            print("Usage: /temp <value> (e.g., /temp 0.8)")
                    
                    elif command.startswith('/length '):
                        try:
                            new_length = int(command.split()[1])
                            if 10 <= new_length <= 1000:
                                max_length = new_length
                                print(f"Max length set to {max_length}")
                            else:
                                print("Length must be between 10 and 1000")
                        except (IndexError, ValueError):
                            print("Usage: /length <value> (e.g., /length 200)")
                    
                    elif command.startswith('/samples '):
                        try:
                            new_samples = int(command.split()[1])
                            if 1 <= new_samples <= 5:
                                num_samples = new_samples
                                print(f"Number of samples set to {num_samples}")
                            else:
                                print("Number of samples must be between 1 and 5")
                        except (IndexError, ValueError):
                            print("Usage: /samples <value> (e.g., /samples 3)")
                    
                    else:
                        print("Unknown command. Type /help for available commands.")
                    
                    continue
                
                # Generate text
                print(f"\nGenerating {num_samples} sample(s) for prompt: '{user_input}'")
                print("-" * 60)
                
                results = self.generate(
                    prompt=user_input,
                    max_length=max_length,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    num_samples=num_samples
                )
                
                for i, text in enumerate(results, 1):
                    if num_samples > 1:
                        print(f"\nSample {i}:")
                    print(text)
                    print("-" * 60)
            
            except KeyboardInterrupt:
                print("\n\nGoodbye!")
                break
            except Exception as e:
                print(f"Error: {e}")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Interactive GPT-2 Text Generator")
    parser.add_argument('--model', type=str, default='best_model.pt',
                       help='Path to trained model checkpoint')
    parser.add_argument('--tokenizer', type=str, default='tokenizer.json',
                       help='Path to tokenizer file')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cpu', 'cuda'],
                       help='Device to use for generation')
    parser.add_argument('--prompt', type=str, default=None,
                       help='Single prompt for generation (non-interactive mode)')
    parser.add_argument('--temperature', type=float, default=0.8,
                       help='Generation temperature')
    parser.add_argument('--max_length', type=int, default=200,
                       help='Maximum generation length')
    parser.add_argument('--num_samples', type=int, default=1,
                       help='Number of samples to generate')
    
    args = parser.parse_args()
    
    # Set seed for reproducible generation
    set_seed(42)
    
    # Check if model files exist
    if not os.path.exists(args.model):
        print(f"Error: Model file '{args.model}' not found!")
        print("Make sure you have trained the model first by running train.py")
        return
    
    if not os.path.exists(args.tokenizer):
        print(f"Error: Tokenizer file '{args.tokenizer}' not found!")
        return
    
    # Create generator
    generator = TextGenerator(args.model, args.tokenizer, args.device)
    
    if args.prompt:
        # Single generation mode
        print(f"Generating text for prompt: '{args.prompt}'")
        results = generator.generate(
            prompt=args.prompt,
            max_length=args.max_length,
            temperature=args.temperature,
            num_samples=args.num_samples
        )
        
        for i, text in enumerate(results, 1):
            if args.num_samples > 1:
                print(f"\nSample {i}:")
            print(text)
    else:
        # Interactive mode
        generator.interactive_mode()


if __name__ == "__main__":
    main()
