from datasets import load_from_disk
from pathlib import Path
from torchinfo import summary
from transformers import (
    GPT2Config,
    GPT2LMHeadModel,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    AutoTokenizer
)

data_dir = Path(__file__).parent.resolve() / "data"
tokenizer_dir = Path(__file__).parent.parent.resolve() / "tokenizer"
output_dir = Path(__file__).parent.parent.resolve() / "foundation-model-512"

dataset = load_from_disk(data_dir)
tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)

context_length = 512
config = GPT2Config(
    vocab_size=len(tokenizer),
    n_positions=context_length,  
    n_ctx=context_length,              
    n_embd=768,                        
    n_layer=12,                       
    n_head=12,                 
    bos_token_id=tokenizer.bos_token_id,
    eos_token_id=tokenizer.eos_token_id,
    pad_token_id=tokenizer.pad_token_id, 
    loss_type="cross_entropy"
)
model = GPT2LMHeadModel(config)
print(model)
summary(model)

tokenizer.pad_token = tokenizer.eos_token
data_collator = DataCollatorForLanguageModeling(tokenizer,mlm=False)

args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    eval_strategy="steps",
    eval_steps=250,
    logging_steps=5,
    gradient_accumulation_steps=8,
    num_train_epochs=5,
    weight_decay=0.1,
    warmup_steps=50,
    lr_scheduler_type="cosine",
    learning_rate=5e-4,
    save_steps=2000,
    fp16=True
)

trainer = Trainer(
    model=model,
    processing_class=tokenizer,
    args=args,
    data_collator=data_collator,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
)
#trainer.train()
trainer.train(resume_from_checkpoint=True)





