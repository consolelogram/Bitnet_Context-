from safetensors import safe_open
import os

path = r'D:\Code\Bitnet\models\bitnet-hf-bf16'
f = [x for x in os.listdir(path) if x.endswith('.safetensors')]
print('Files:', f)

with safe_open(os.path.join(path, f[0]), framework='pt') as st:
    keys = list(st.keys())
    print(f'Tensors: {len(keys)}')
    for k in keys[:5]:
        t = st.get_tensor(k)
        print(f'  {k}: {t.dtype} {tuple(t.shape)}')