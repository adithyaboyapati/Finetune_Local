from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch

BASE_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
LORA_MODEL = "./tinyllama_lora"

device = "mps"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(LORA_MODEL)

# Load base model
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float32,
    device_map={"": "mps"}
)

# Load LoRA adapters
model = PeftModel.from_pretrained(base_model, LORA_MODEL)

model.to(device)

prompt = "Who is Adithya?"

inputs = tokenizer(prompt, return_tensors="pt").to(device)

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=50
    )

response = tokenizer.decode(outputs[0], skip_special_tokens=True)

print(response)