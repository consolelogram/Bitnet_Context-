from huggingface_hub import snapshot_download

snapshot_download(

    repo_id="microsoft/bitnet-b1.58-2B-4T-bf16",

    local_dir=r"D:\Code\Bitnet\models\bitnet-hf-bf16",

)

print("Done!")