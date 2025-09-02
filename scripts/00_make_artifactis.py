# 00_make_artifacts.py  (Python 3.11)
import numpy as np
rng = np.random.default_rng(123)

N, C, H, W = 16, 3, 32, 32
OC, IC, KH, KW = 8, 3, 3, 3
OUT = 16

from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))
ROOT_DIR = Path(__file__).resolve().parent.parent
RESULT_DIR = ROOT_DIR / "results"
RESULT_DIR.mkdir(parents=True, exist_ok=True)

x = rng.standard_normal((N, C, H, W), dtype=np.float32)

# Conv weight/bias (OIHW)
w_conv = rng.standard_normal((OC, IC, KH, KW), dtype=np.float32)
b_conv = rng.standard_normal((OC,), dtype=np.float32)

# Linear weight/bias: PyTorch의 Linear는 (out_features, in_features)
in_lin = OC * (H - KH + 1) * (W - KW + 1)  # 8*30*30
w_lin = rng.standard_normal((OUT, in_lin), dtype=np.float32)
b_lin = rng.standard_normal((OUT,), dtype=np.float32)

np.savez(f"{RESULT_DIR}/weights_inputs.npz",
         x=x, w_conv=w_conv, b_conv=b_conv, w_lin=w_lin, b_lin=b_lin)
print(f"saved → {RESULT_DIR}/weights_inputs.npz")

