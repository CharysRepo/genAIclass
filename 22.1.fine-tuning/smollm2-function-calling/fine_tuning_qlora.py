import os
os.environ["HF_HOME"] = "F:/hf"
from datasets import load_from_disk
import torch
from pathlib import Path
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig
)
from trl import (
    SFTTrainer,
    SFTConfig
)
from peft import LoraConfig, TaskType, prepare_model_for_kbit_training
from torchinfo import summary

data_dir = Path(__file__).parent.resolve() / "data"
model_dir = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
output_dir = Path(__file__).parent.resolve() / "smollm2-1.7b-it-function-calling-adapter-qlora"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16, 
    bnb_4bit_use_double_quant=True
)

dataset = load_from_disk(data_dir)
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForCausalLM.from_pretrained(model_dir, 
                                             quantization_config=bnb_config, 
                                             device_map="auto")
model = prepare_model_for_kbit_training(model)
print(model)
summary(model)

rank_dimension = 16
# lora_alpha: scaling factor for LoRA layers (higher = stronger adaptation)
lora_alpha = 64
lora_dropout = 0.05
peft_config = LoraConfig(r=rank_dimension,
                         lora_alpha=lora_alpha,
                         lora_dropout=lora_dropout,
                         target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                         task_type=TaskType.CAUSAL_LM)

training_args = SFTConfig(
    output_dir=output_dir,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    eval_strategy="steps",
    eval_steps=10,
    logging_steps=1,
    gradient_accumulation_steps=8,
    num_train_epochs=3,
    weight_decay=0.1,
    warmup_steps=2,
    lr_scheduler_type="cosine",
    learning_rate=5e-4,
    save_steps=5,
    fp16=True,
    assistant_only_loss=False
)

trainer = SFTTrainer(
    model=model,
    processing_class=tokenizer,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    peft_config=peft_config
)

trainer.train()
#trainer.train(resume_from_checkpoint=True)


