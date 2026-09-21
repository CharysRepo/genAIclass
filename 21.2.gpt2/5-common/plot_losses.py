import json
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def plot_losses(model_dir):
    state_path = model_dir / "trainer_state.json"
    with open(state_path, "r") as f:
        data = json.load(f)

    # Extract log history logs
    log_history = data.get("log_history", [])

    # Parse and separate training loss and validation loss
    train_data = [log for log in log_history if "loss" in log]
    eval_data = [log for log in log_history if "eval_loss" in log]

    df_train = pd.DataFrame(train_data)
    df_eval = pd.DataFrame(eval_data)

    # Plot the results
    plt.figure(figsize=(10, 6))

    if not df_train.empty:
        plt.plot(df_train["step"], df_train["loss"], label="Training Loss", color="blue", marker="o")
        
    if not df_eval.empty:
        plt.plot(df_eval["step"], df_eval["eval_loss"], label="Validation Loss", color="red", marker="x")

    plt.title("Hugging Face Trainer: Training vs Validation Loss Curves")
    plt.xlabel("Steps")
    plt.ylabel("Loss")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.show()

if __name__ == "__main__":
    model_dir = Path(__file__).parent.parent.resolve() / "foundation-model-512" / "checkpoint-14285"
    #model_dir = Path(__file__).parent.parent.resolve() / "it-model-512" / "checkpoint-708"
    plot_losses(model_dir)
