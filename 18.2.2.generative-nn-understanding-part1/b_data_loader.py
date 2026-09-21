from torch.utils.data import Dataset, DataLoader
import torch
from a_tokenizer import SimpleTokenizer

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
    
if __name__ == "__main__":
    CONTEXT_SIZE = 5

    raw_train_text = "i love deep learning and i love building neural networks"
    raw_val_text   = "i love python code and i love learning"
    
    tokenizer = SimpleTokenizer(raw_train_text) 
    train_dataset = CustomDataset(raw_train_text, tokenizer, context_length=CONTEXT_SIZE) 
    val_dataset = CustomDataset(raw_val_text, tokenizer, context_length=CONTEXT_SIZE) 

    print("Train dataset")
    print(raw_train_text)
    print(tokenizer.encode(raw_train_text))
    for item in train_dataset:
        print(item)

    print("Val dataset")
    print(raw_val_text)
    print(tokenizer.encode(raw_val_text))
    for item in val_dataset:
        print(item)

    train_loader = DataLoader(dataset=train_dataset, batch_size=2, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=2, shuffle=False)

    print("Train dataloader")
    for item in train_loader:
        print(item)

    print("Val dataloader")
    for item in val_loader:
        print(item)