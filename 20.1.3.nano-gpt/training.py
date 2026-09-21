from pathlib import Path
from tqdm import tqdm
from model import NanoGPT
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator
import torch.optim as optim
from data_loader import SimpleTokenizer, CustomDataset
import pickle
from torchinfo import summary


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
    optimizer = optim.Adam(model.parameters(), lr=5e-4, weight_decay=0.01)

    train_losses = []
    val_losses = []

    total_steps = epochs * len(train_loader)
    progress_bar = tqdm(total=total_steps, desc="Training Steps", unit="step")

    for epoch in range(epochs):
        train_loss = 0.0
        model.train()
        for input_batch, target_batch in train_loader:
            input_batch, target_batch = input_batch.to(device), target_batch.to(device)

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
            
            progress_bar.update(1)
            progress_bar.set_postfix({
                "epoch": f"{epoch+1}/{epochs}",
                "step_loss": f"{loss.item():.4f}"
            })

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

        tqdm.write(f" [Epoch {epoch+1} Complete] Avg Train Loss: {train_loss:.4f} | Avg Val Loss: {val_loss:.4f}")
        
    return train_losses, val_losses

if __name__ == "__main__":
    CONTEXT_SIZE = 64
    
    # prepare the dataset and tokenizer
    current_dir = Path(__file__).resolve().parent / "data"
    input_path = current_dir / "input.txt"
    with open(input_path, 'r', encoding='utf-8') as f:
        raw_input_text = f.read()
    tokenizer = SimpleTokenizer(raw_input_text) 
    print(f"Tokenizer created with vocab size: {tokenizer.vocab_size}")
    
    train_path = current_dir / "train.txt"
    with open(train_path, 'r', encoding='utf-8') as f:
        raw_train_text = f.read()
    train_dataset = CustomDataset(raw_train_text, tokenizer, context_length=CONTEXT_SIZE) 
    print(f"Training dataset size: {len(train_dataset)} samples")

    val_path = current_dir / "val.txt"
    with open(val_path, 'r', encoding='utf-8') as f:
        raw_val_text = f.read()
    val_dataset = CustomDataset(raw_val_text, tokenizer, context_length=CONTEXT_SIZE) 
    print(f"Validation dataset size: {len(val_dataset)} samples")
    
    #define model configuration
    config = {
        "vocab_size": tokenizer.vocab_size,
        "context_length": CONTEXT_SIZE,
        "emb_dim": 256, #128
        "hidden_dim": 512, #64
        "n_heads": 8, #4
        "drop_rate": 0.1, #0.25
    }
    model = NanoGPT(config)
    print(model)
    summary(model)

    # train the model
    batch_size = 32
    epochs = 5
    train_losses, val_losses = train(model, epochs, batch_size, train_dataset, val_dataset)
    plot_losses(list(range(1, epochs + 1)), train_losses, val_losses)

    #save the model and tokenizer
    model_dir = Path(__file__).resolve().parent / "models"
    model_dir.mkdir(exist_ok=True)
    model_path = model_dir / "generative_model_v1.pt"
    tokenizer_path = model_dir / "tokenizer.pkl" 
    torch.save(model, model_path)
    print(f"Model saved to {model_path}")
    with open(tokenizer_path, 'wb') as f_tok:
        pickle.dump(tokenizer, f_tok)
    print(f"Tokenizer saved to {tokenizer_path}")

