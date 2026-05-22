from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer
import torch

MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# -----------------------------
# Load tokenizer
# -----------------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

# -----------------------------
# QLoRA (4-bit quantization) configuration
# IMPORTANT NOTE for macOS (MPS) Users:
# The `bitsandbytes` library (which powers QLoRA 4-bit quantization) traditionally 
# requires a CUDA GPU. Running 4-bit quantization via bitsandbytes on macOS MPS 
# may result in errors or require specialized forks.
# -----------------------------
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
)

# -----------------------------
# Load quantized model
# -----------------------------
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto"
)

# Prepare model for k-bit training (casts layers, adds hooks)
model = prepare_model_for_kbit_training(model)

# -----------------------------
# LoRA configuration
# -----------------------------
# QLoRA often adapts all linear modules for optimal performance
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
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
    output_dir="./outputs_qlora",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=2,
    learning_rate=2e-4,
    num_train_epochs=3,
    logging_steps=1,
    save_strategy="epoch",
    # Mixed precision flags depending on CUDA availability
    fp16=not torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False,
    bf16=torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False,
    optim="paged_adamw_8bit" if torch.cuda.is_available() else "adamw_torch"
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
model.save_pretrained("tinyllama_qlora")
tokenizer.save_pretrained("tinyllama_qlora")

print("QLoRA Training complete!")
