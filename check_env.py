import torch
import transformers
import safetensors
import numpy
import json

print("torch:", torch.__version__)
print("transformers:", transformers.__version__)
print("safetensors:", safetensors.__version__)
print("numpy:", numpy.__version__)

cfg = json.load(open(r"D:\Code\Bitnet\models\bitnet-hf\config.json"))
print("\nModel type:", cfg.get("model_type"))
print("Architecture:", cfg.get("architectures"))
print("Hidden size:", cfg.get("hidden_size"))
print("RoPE theta:", cfg.get("rope_theta"))
print("Max position embeddings:", cfg.get("max_position_embeddings"))