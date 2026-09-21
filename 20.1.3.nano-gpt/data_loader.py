import torch
from torch.utils.data import Dataset
import re

class SimpleTokenizer:
    def __init__(self, text):
        self.words = self._tokenize(text)
        self.vocab =  ["<UNK>"] + sorted(list(set(self.words)))
        self.vocab_size = len(self.vocab)

        self.word_to_idx = {word: idx for idx, word in enumerate(self.vocab)}
        self.idx_to_word = {idx: word for idx, word in enumerate(self.vocab)}

    def _tokenize(self, text_string):
        cleaned = text_string.lower()
        cleaned = re.sub(r"([.,!?])", r" \1 ", cleaned)
        return cleaned.split()

    def encode(self, text_string):
        cleaned_tokens = self._tokenize(text_string)
        return [self.word_to_idx.get(w, self.word_to_idx["<UNK>"]) for w in cleaned_tokens]

    def decode(self, token_ids):
        return " ".join([self.idx_to_word[idx] for idx in token_ids])
    
class CustomDataset(Dataset): 
    def __init__(self, text, tokenizer, context_length): 
        self.tokens = tokenizer.encode(text) 
        self.context_length = context_length
        self.inputs = []
        self.targets = []
        
        for i in range(len(self.tokens) - context_length):
            self.inputs.append(self.tokens[i : i + context_length])
            self.targets.append(self.tokens[i + context_length])

    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, idx):
        return (
            torch.tensor(self.inputs[idx], dtype=torch.long),
            torch.tensor(self.targets[idx], dtype=torch.long)
        )

