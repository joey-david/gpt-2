"""
Character-level tokenizer for GPT-2 implementation
"""
import json
import pickle
from typing import List, Dict, Optional


class CharacterTokenizer:
    """Simple character-level tokenizer"""
    
    def __init__(self):
        self.char_to_id: Dict[str, int] = {}
        self.id_to_char: Dict[int, str] = {}
        self.vocab_size = 0
        
    def build_vocab(self, text: str) -> None:
        """Build vocabulary from text"""
        chars = sorted(list(set(text)))
        self.vocab_size = len(chars)
        
        self.char_to_id = {ch: i for i, ch in enumerate(chars)}
        self.id_to_char = {i: ch for i, ch in enumerate(chars)}
        
        print(f"Vocabulary size: {self.vocab_size}")
        print(f"Sample characters: {chars[:20]}...")
        
    def encode(self, text: str) -> List[int]:
        """Encode text to list of token ids"""
        return [self.char_to_id[ch] for ch in text if ch in self.char_to_id]
    
    def decode(self, token_ids: List[int]) -> str:
        """Decode list of token ids to text"""
        return ''.join([self.id_to_char[id] for id in token_ids if id in self.id_to_char])
    
    def save(self, filepath: str) -> None:
        """Save tokenizer to file"""
        data = {
            'char_to_id': self.char_to_id,
            'id_to_char': self.id_to_char,
            'vocab_size': self.vocab_size
        }
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    def load(self, filepath: str) -> None:
        """Load tokenizer from file"""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.char_to_id = data['char_to_id']
        # Convert string keys back to integers for id_to_char
        self.id_to_char = {int(k): v for k, v in data['id_to_char'].items()}
        self.vocab_size = data['vocab_size']


class BPETokenizer:
    """Byte Pair Encoding tokenizer (more advanced option)"""
    
    def __init__(self, vocab_size: int = 1000):
        self.vocab_size = vocab_size
        self.vocab: Dict[str, int] = {}
        self.merges: List[tuple] = []
        
    def get_stats(self, vocab: Dict[str, int]) -> Dict[tuple, int]:
        """Get frequency of consecutive symbol pairs"""
        pairs = {}
        for word, freq in vocab.items():
            symbols = word.split()
            for i in range(len(symbols) - 1):
                pair = (symbols[i], symbols[i + 1])
                pairs[pair] = pairs.get(pair, 0) + freq
        return pairs
    
    def merge_vocab(self, pair: tuple, vocab: Dict[str, int]) -> Dict[str, int]:
        """Merge the most frequent pair in vocabulary"""
        new_vocab = {}
        bigram = ' '.join(pair)
        replacement = ''.join(pair)
        
        for word in vocab:
            new_word = word.replace(bigram, replacement)
            new_vocab[new_word] = vocab[word]
        return new_vocab
    
    def build_vocab(self, text: str) -> None:
        """Build BPE vocabulary from text"""
        # Initialize vocabulary with character-level splits
        words = text.split()
        vocab = {}
        for word in words:
            word = ' '.join(list(word)) + ' </w>'
            vocab[word] = vocab.get(word, 0) + 1
        
        # Perform BPE merges
        for i in range(self.vocab_size - len(set(text))):
            pairs = self.get_stats(vocab)
            if not pairs:
                break
            
            best_pair = max(pairs, key=pairs.get)
            vocab = self.merge_vocab(best_pair, vocab)
            self.merges.append(best_pair)
        
        # Create final vocabulary
        self.vocab = {}
        for word, freq in vocab.items():
            for symbol in word.split():
                if symbol not in self.vocab:
                    self.vocab[symbol] = len(self.vocab)
        
        print(f"BPE Vocabulary size: {len(self.vocab)}")
    
    def encode(self, text: str) -> List[int]:
        """Encode text using BPE"""
        # Simplified encoding - in practice this would be more complex
        tokens = []
        for char in text:
            if char in self.vocab:
                tokens.append(self.vocab[char])
            else:
                tokens.append(self.vocab.get('<unk>', 0))
        return tokens
    
    def decode(self, token_ids: List[int]) -> str:
        """Decode BPE tokens to text"""
        id_to_token = {v: k for k, v in self.vocab.items()}
        tokens = [id_to_token.get(id, '<unk>') for id in token_ids]
        return ''.join(tokens).replace('</w>', ' ')
