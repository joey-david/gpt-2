import model
import os
from typing import Union, Tuple

# import text, convert to string
with open(file="charwise-gpt/dickens/combined.txt", mode='r', encoding='utf-8') as f:
    text = f.read()

# tokenize (by character)
chars = sorted(list(set(text)))
print(f"Vocab size: {len(chars)}")
print(f"Vocabulary : {chars}")

