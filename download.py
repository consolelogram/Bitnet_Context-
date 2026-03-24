# test_load.py
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_path = r"D:\Code\Bitnet\models\bitnet-hf"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_path)

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float32,
    device_map="cpu",
    output_attentions=False
)
print("Model loaded.")
print("Parameters:", sum(p.numel() for p in model.parameters()))