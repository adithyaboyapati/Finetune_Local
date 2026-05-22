from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer
import torch

MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# -----------------------------
# Load tokenizer
# -----------------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

tokenizer.pad_token = tokenizer.eos_token

# -----------------------------
# Load model
# -----------------------------
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float32,
    device_map="auto"
)

# -----------------------------
# LoRA configuration
# -----------------------------
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# Inject LoRA adapters
model = get_peft_model(model, lora_config)

# -----------------------------
# Load dataset
# -----------------------------
dataset = load_dataset("json", data_files="dataset.json")

# -----------------------------
# Formatting function
# -----------------------------
def format_example(example):
    text = f"""
### Instruction:
{example['instruction']}

### Response:
{example['response']}
"""
    return {"text": text}

dataset = dataset.map(format_example)

# -----------------------------
# Training arguments
# -----------------------------
training_args = TrainingArguments(
    output_dir="./outputs",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=2,
    learning_rate=2e-4,
    num_train_epochs=3,
    logging_steps=1,
    save_strategy="epoch",
    fp16=False,
)

# -----------------------------
# Trainer
# -----------------------------
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset["train"],
    args=training_args,
    dataset_text_field="text",
)

# -----------------------------
# Start training
# -----------------------------
trainer.train()

# -----------------------------
# Save LoRA adapters
# -----------------------------
model.save_pretrained("tinyllama_lora")
tokenizer.save_pretrained("tinyllama_lora")

print("Training complete!")