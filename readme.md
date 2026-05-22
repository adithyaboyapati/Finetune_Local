# TinyLlama LoRA / QLoRA Fine-Tuning

Fine-tune [TinyLlama-1.1B-Chat](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0) on a custom instruction dataset using **LoRA** (standard) or **QLoRA** (4-bit quantized). Includes a simple inference script to test the trained adapter locally.

---

## Project Structure

```
.
├── train.py              # LoRA fine-tuning (CPU / MPS / CUDA)
├── trainer_qlora.py      # QLoRA fine-tuning (requires CUDA for 4-bit quantization)
├── inference.py          # Run inference with the trained LoRA adapter
├── dataset.json          # Custom instruction-response training data
├── requirements.txt      # Python dependencies
├── tinyllama_lora/       # Saved LoRA adapter (after training)
└── outputs/              # Training checkpoints
```

---

## Setup

```bash
pip install -r requirements.txt
```

> **macOS (Apple Silicon):** The `bitsandbytes` library used for QLoRA 4-bit quantization requires CUDA. Use `train.py` (LoRA) on MPS-based Macs instead of `trainer_qlora.py`.

---

## Dataset Format

`dataset.json` uses an instruction–response format:

```json
[
  {
    "instruction": "Who is Adithya?",
    "response": "Adithya is a Data Scientist specializing in GenAI, NLP, and LLM systems."
  }
]
```

Add as many examples as needed following the same schema.

---

## Training

### LoRA (standard — works on CPU, MPS, CUDA)

```bash
python train.py
```

Trains with `r=8`, `lora_alpha=16`, targeting `q_proj` and `v_proj`. Saves the adapter to `tinyllama_lora/`.

| Hyperparameter | Value |
|---|---|
| Learning rate | 2e-4 |
| Batch size | 1 |
| Gradient accumulation | 2 steps |
| Epochs | 3 |
| Precision | float32 |

### QLoRA (4-bit quantized — requires CUDA)

```bash
python trainer_qlora.py
```

Uses NF4 quantization via `BitsAndBytesConfig`. Targets all major linear layers (`q_proj`, `v_proj`, `k_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`). Saves the adapter to `tinyllama_qlora/`.

| Hyperparameter | Value |
|---|---|
| Quantization | 4-bit NF4 + double quant |
| Learning rate | 2e-4 |
| Batch size | 1 |
| Gradient accumulation | 2 steps |
| Epochs | 3 |
| Optimizer | paged_adamw_8bit (CUDA) / adamw_torch (CPU) |

---

## Inference

```bash
python inference.py
```

Loads the base model and merges the LoRA adapter from `tinyllama_lora/`, then runs generation on a sample prompt. Configured for **MPS** by default — change `device = "mps"` to `"cuda"` or `"cpu"` as needed.

---

## Key Dependencies

| Package | Version |
|---|---|
| `transformers` | 4.41.2 |
| `peft` | 0.11.1 |
| `trl` | 0.8.6 |
| `torch` | 2.12.0 |
| `datasets` | 4.8.5 |
| `bitsandbytes` | 0.49.2 |
| `accelerate` | 0.31.0 |
