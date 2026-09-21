from datasets import load_from_disk
import torch
from pathlib import Path
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM
)
from trl import (
    setup_chat_format,
    DPOTrainer,
    DPOConfig
)

data_dir = Path(__file__).parent.resolve() / "data"
tokenizer_dir = Path(__file__).parent.parent.resolve() / "tokenizer"
model_dir = Path(__file__).parent.parent.resolve() / "it-model-512" / "checkpoint-780"
output_dir = Path(__file__).parent.parent.resolve() / "pt-model-512"

device = "cuda" if torch.cuda.is_available() else "cpu"

dataset = load_from_disk(data_dir)
tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
model = AutoModelForCausalLM.from_pretrained(model_dir).to(device)
model, tokenizer = setup_chat_format(model, tokenizer, format="chatml")

training_args = DPOConfig(
    output_dir=output_dir,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    eval_strategy="steps",
    eval_steps=100,
    logging_steps=1,
    gradient_accumulation_steps=8,
    num_train_epochs=3,
    weight_decay=0.1,
    warmup_steps=10,
    lr_scheduler_type="cosine",
    learning_rate=5e-4,
    save_steps=100,
    fp16=True,
    completion_only_loss=True
)

trainer = DPOTrainer(
    model=model,
    processing_class=tokenizer,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"]
)

trainer.train()
#trainer.train(resume_from_checkpoint=True)


