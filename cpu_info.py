import subprocess
import platform

print("Platform:", platform.processor())

# check via cpuinfo
try:
    import cpuinfo
    info = cpuinfo.get_cpu_info()
    print("CPU:", info['brand_raw'])
    print("Flags:", [f for f in info.get('flags', []) if f in ['avx', 'avx2', 'avx512f', 'sse4_1', 'sse4_2', 'neon']])
except ImportError:
    print("Installing cpuinfo...")
    subprocess.run(['pip', 'install', 'py-cpuinfo'])
    print("Run this script again.")