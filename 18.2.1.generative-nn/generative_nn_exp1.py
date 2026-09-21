import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator
import torch.optim as optim
import re
from torchinfo import summary

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


class GenerativeNetwork(nn.Module):
    def __init__(self, vocab_size, emb_dim):
        super().__init__()
        self.tok_embedding = nn.Embedding(vocab_size, emb_dim)
        self.output_layer = nn.Linear(emb_dim, vocab_size)

    def forward(self, x):
        #print(x, x.shape)
        x = self.tok_embedding(x)
        #print(x, x.shape)
        x = self.output_layer(x)
        #print(x, x.shape)
        if x.shape[1] == 1:
            x = x.squeeze(1) 
        else:
            x = x[:, -1, :] 
        return x

def plot_losses(epochs, train_losses, val_losses):
    fig, ax1 = plt.subplots(figsize=(10, 10))

    ax1.plot(epochs, train_losses, label="Training loss")
    ax1.plot(epochs, val_losses, linestyle="-.", label="Validation loss")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend(loc="upper right")
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))

    fig.tight_layout()
    plt.show()
    plt.close()

def train(model, epochs, batch_size, train_dataset, val_dataset):
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
   
    #optimizer = torch.optim.SGD(model.parameters(), lr=0.1) 
    optimizer = optim.Adam(model.parameters(), lr=0.05)

    train_losses = []
    val_losses = []

    step = 1
    for epoch in range(epochs):
        train_loss = 0.0
        model.train()
        print(f"Epoch {epoch+1}/{epochs}")
        for input_batch, target_batch in train_loader:
            input_batch, target_batch = input_batch.to(device), target_batch.to(device)
            print(f"step number:{step}")
            print(input_batch, target_batch)

            # clear gradients of parameters from previous step
            optimizer.zero_grad() 

            # compute neural net output (forward pass)
            output = model(input_batch)

            # comput loss
            loss = F.cross_entropy(output, target_batch)            
            train_loss += loss.item()

            # compute the slope of loss curve at this weights (backward pass)
            loss.backward()

            # update parameters based on learning rate and calculated slopes
            optimizer.step()
            print(model.state_dict())
            #input()
            step += 1

        val_loss = 0.0
        model.eval()        
        with torch.no_grad():
            for input_batch, target_batch in val_loader:
                input_batch, target_batch = input_batch.to(device), target_batch.to(device)
                output = model(input_batch)
                val_loss += F.cross_entropy(output, target_batch).item()

        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(f"epoch: {epoch+1}, train_loss: {train_loss:.4f}, val_loss: {val_loss:.4f}")
        
    return train_losses, val_losses

def inference(model, tokenizer, start_phrase, context_length, max_gen_length):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval() 
    
    generated_tokens = tokenizer.encode(start_phrase)    
    print(generated_tokens)
    for _ in range(max_gen_length):
        context_tokens = generated_tokens[-context_length:] 
        input_tensor = torch.tensor([context_tokens], dtype=torch.long).to(device) 
 
        with torch.no_grad():
            print(input_tensor)
            logits = model(input_tensor)
            print(logits)
            predicted_idx = torch.argmax(logits, dim=-1).item() 
            print(predicted_idx)
            print(tokenizer.idx_to_word[predicted_idx])
            generated_tokens.append(predicted_idx)
        
    print(generated_tokens)
    return tokenizer.decode(generated_tokens)


if __name__ == "__main__":
    
    raw_train_text = """
    Artificial intelligence and deep learning are transforming how we build technology.
    I love deep learning and I love building neural networks from scratch.
    Neural networks use layers of linear math and activation functions to learn.
    We write clean python code to process training datasets and compute loss parameters.
    An optimization algorithm updates weights based on calculated learning rate slopes.
    Models can extract features from text and generate new phrases over time.
    Building deep neural models requires computing clean loss calculations regularly.
    Training networks on complex structures can optimize predictive matrix logic.
    """ 
    
    # Expanded validation text block
    raw_val_text = """
    Deep learning algorithms can transform regular technology pipelines.
    I love python code and I love learning about advanced neural networks.
    A simple optimization framework adjusts parameters and layers to lower loss values.
    Models require clean functions to generate coherent text outputs.
    """

    CONTEXT_SIZE = 5
    
    tokenizer = SimpleTokenizer(raw_train_text) 
    train_dataset = CustomDataset(raw_train_text, tokenizer, context_length=CONTEXT_SIZE) 
    val_dataset = CustomDataset(raw_val_text, tokenizer, context_length=CONTEXT_SIZE) 
    
    print(tokenizer.vocab_size)
    model = GenerativeNetwork(tokenizer.vocab_size, emb_dim=32)
    batch_size = 16
    #len(train_dataset)
    epochs = 50

    print(len(train_dataset), len(val_dataset))
    train_losses, val_losses = train(model, epochs, batch_size, train_dataset, val_dataset)
    print(model)
    summary(model)
    print(model.state_dict())
    plot_losses(list(range(1, epochs + 1)), train_losses, val_losses)

    while True:
        start_text = input(">>")
        if start_text in ["exit", "quit"]:
            break
        output_text = inference(model, tokenizer, start_phrase=start_text, context_length = CONTEXT_SIZE, max_gen_length=10)
        print("Generated Text:", output_text)

# we write clean python code to
# Generated Text: we write clean python code to process training networks on complex structures can extract features from
# i love deep learning and
#Generated Text: i love deep learning and compute loss calculations regularly . training networks on complex structures
# deep neural networks
# Generated Text: deep neural networks on complex structures can extract features from text and compute