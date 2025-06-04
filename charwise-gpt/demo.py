"""
Demo script to test the GPT-2 implementation
"""
import torch
from model import GPT2Model, GPT2Config
from tokenizer import CharacterTokenizer
from data_utils import DataProcessor
import os


def test_tokenizer():
    """Test the tokenizer implementation"""
    print("Testing tokenizer...")
    
    # Sample text
    text = "Hello, this is a test of the tokenizer implementation!"
    
    # Create tokenizer
    tokenizer = CharacterTokenizer()
    tokenizer.build_vocab(text)
    
    # Test encoding/decoding
    encoded = tokenizer.encode(text)
    decoded = tokenizer.decode(encoded)
    
    print(f"Original: {text}")
    print(f"Encoded: {encoded}")
    print(f"Decoded: {decoded}")
    print(f"Round-trip successful: {text == decoded}")
    print()


def test_model():
    """Test the model implementation"""
    print("Testing model...")
    
    # Create small model for testing
    config = GPT2Config(
        vocab_size=100,
        d_model=128,
        n_heads=4,
        n_layers=2,
        d_ff=512,
        max_seq_len=64,
        dropout=0.1
    )
    
    model = GPT2Model(
        vocab_size=config.vocab_size,
        d_model=config.d_model,
        n_heads=config.n_heads,
        n_layers=config.n_layers,
        d_ff=config.d_ff,
        max_seq_len=config.max_seq_len,
        dropout=config.dropout
    )
    
    # Test forward pass
    batch_size = 2
    seq_len = 32
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    
    with torch.no_grad():
        logits = model(input_ids)
    
    print(f"Input shape: {input_ids.shape}")
    print(f"Output shape: {logits.shape}")
    print(f"Expected output shape: {(batch_size, seq_len, config.vocab_size)}")
    print(f"Shape test passed: {logits.shape == (batch_size, seq_len, config.vocab_size)}")
    
    # Test generation
    print("\nTesting generation...")
    prompt = torch.randint(0, config.vocab_size, (1, 10))
    generated = model.generate(prompt, max_length=20, temperature=1.0)
    
    print(f"Prompt shape: {prompt.shape}")
    print(f"Generated shape: {generated.shape}")
    print(f"Generation successful: {generated.shape[1] > prompt.shape[1]}")
    print()


def test_data_processing():
    """Test data processing utilities"""
    print("Testing data processing...")
    
    # Load sample text
    if os.path.exists("charwise-gpt/dickens/combined.txt"):
        with open("charwise-gpt/dickens/combined.txt", 'r', encoding='utf-8') as f:
            raw_text = f.read()[:10000]  # First 10k chars for testing
        
        processor = DataProcessor()
        cleaned_text = processor.clean_text(raw_text)
        stats = processor.get_text_statistics(cleaned_text)
        
        print(f"Raw text length: {len(raw_text)}")
        print(f"Cleaned text length: {len(cleaned_text)}")
        print(f"Unique characters: {stats['unique_chars']}")
        print(f"Sample characters: {stats['vocab'][:20]}")
        print()
    else:
        print("Dickens text file not found, skipping data processing test")
        print()


def test_full_pipeline():
    """Test the complete pipeline with a tiny model"""
    print("Testing full pipeline...")
    
    # Sample text for training
    sample_text = """
    It was the best of times, it was the worst of times.
    It was a bright cold day in April, and the clocks were striking thirteen.
    In a hole in the ground there lived a hobbit.
    Call me Ishmael. Some years ago—never mind how long precisely.
    """ * 10  # Repeat to have enough data
    
    # Create tokenizer
    tokenizer = CharacterTokenizer()
    tokenizer.build_vocab(sample_text)
    
    print(f"Vocabulary size: {tokenizer.vocab_size}")
    
    # Create tiny model
    model = GPT2Model(
        vocab_size=tokenizer.vocab_size,
        d_model=64,
        n_heads=2,
        n_layers=2,
        d_ff=256,
        max_seq_len=128,
        dropout=0.1
    )
    
    # Test generation before training
    print("\nGeneration before training:")
    prompt = "It was"
    encoded_prompt = torch.tensor([tokenizer.encode(prompt)])
    
    with torch.no_grad():
        generated = model.generate(encoded_prompt, max_length=50, temperature=1.0)
        generated_text = tokenizer.decode(generated[0].tolist())
        print(f"Generated: {generated_text}")
    
    # Simple training step
    print("\nTesting training step...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    
    # Prepare training data
    tokens = tokenizer.encode(sample_text)
    block_size = 32
    
    for i in range(5):  # Just a few training steps
        # Random batch
        start_idx = torch.randint(0, len(tokens) - block_size, (1,)).item()
        input_ids = torch.tensor([tokens[start_idx:start_idx + block_size]])
        targets = torch.tensor([tokens[start_idx + 1:start_idx + block_size + 1]])
        
        # Forward pass
        logits = model(input_ids)
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.view(-1)
        )
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if i % 2 == 0:
            print(f"Step {i}, Loss: {loss.item():.4f}")
    
    # Test generation after training
    print("\nGeneration after training:")
    with torch.no_grad():
        generated = model.generate(encoded_prompt, max_length=100, temperature=0.8)
        generated_text = tokenizer.decode(generated[0].tolist())
        print(f"Generated: {generated_text}")
    
    print("Full pipeline test completed!")


def main():
    """Run all tests"""
    print("=" * 60)
    print("GPT-2 IMPLEMENTATION TESTING")
    print("=" * 60)
    
    try:
        test_tokenizer()
        test_model()
        test_data_processing()
        test_full_pipeline()
        
        print("=" * 60)
        print("ALL TESTS PASSED! ✅")
        print("=" * 60)
        print("\nYou can now run the full training with:")
        print("python train.py")
        print("\nOr try interactive generation with:")
        print("python generate.py --prompt 'It was the best of times'")
        
    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
