import subprocess
import sys

model_dir = r"D:\Code\Bitnet\models\bitnet-hf"
outfile = r"D:\Code\Bitnet\BitNet\models\bitnet-b1.58-2B-4T\ggml-model-tl1.gguf"

cmd = [
    sys.executable, "utils/convert-hf-to-gguf-bitnet.py",
    model_dir,
    "--outfile", outfile,
    "--outtype", "tl1"
]

print("Running conversion:", " ".join(cmd))
result = subprocess.run(cmd, capture_output=False)

if result.returncode != 0:
    print("Conversion failed")
else:
    print("Done. GGUF at:", outfile)