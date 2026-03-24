from huggingface_hub import hf_hub_download

path = hf_hub_download(
    repo_id="microsoft/bitnet-b1.58-2B-4T",
    filename="ggml-model-i2_s.gguf",
    local_dir=r"D:\Code\Bitnet\BitNet\models\bitnet-b1.58-2B-4T"
)
print("Downloaded to:", path)