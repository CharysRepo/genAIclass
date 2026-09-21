import re

class SimpleTokenizer:
    def __init__(self, text):
        self.words = self._tokenize(text)
        self.vocab =  ["<UNK>"] + sorted(list(set(self.words)))
        self.vocab_size = len(self.vocab)
        
        # Bi-directional mapping dictionaries
        self.word_to_idx = {word: idx for idx, word in enumerate(self.vocab)}
        self.idx_to_word = {idx: word for idx, word in enumerate(self.vocab)}

    def _tokenize(self, text_string):
        cleaned = text_string.lower()
        cleaned = re.split(r'([,.?_!"()\']|--|\s)', cleaned)
        cleaned = [item for item in cleaned if item != ' ']
        return cleaned

    def encode(self, text_string):
        cleaned_tokens = self._tokenize(text_string)
        return [self.word_to_idx.get(w, self.word_to_idx["<UNK>"]) for w in cleaned_tokens]

    def decode(self, token_ids):
        text = " ".join([self.idx_to_word[idx] for idx in token_ids])
        text = re.sub(r'\s+([,.?!"()\'])', r"\1", text)
        return text
    
if __name__ == "__main__":
    tokenizer = SimpleTokenizer("i love deep learning and i love building neural networks")
    print(tokenizer.vocab)
    print(tokenizer.vocab_size)
    print(tokenizer.word_to_idx)
    print(tokenizer.idx_to_word)

    text = "i love neural networks"
    ids = tokenizer.encode(text)
    print(ids)
    print(tokenizer.decode(ids))
